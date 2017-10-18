/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

    typedef unsigned short VertexIndex;
    typedef glm::vec3 VertexAttributePosition;
    typedef glm::vec3 VertexAttributeNormal;
    typedef glm::vec2 VertexAttributeTexcoord;
    typedef unsigned char TextureData;

    typedef unsigned char BufferByte;

    enum PrimitiveType {
        Point = 1,
        Line = 2,
        Triangle = 3
    };

    struct VertexOut {
        glm::vec4 pos;

        // TODO: add new attributes to your VertexOut
        // The attributes listed below might be useful, 
        // but always feel free to modify on your own

        glm::vec3 eyePos;	// eye space position used for shading
        glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
        glm::vec3 col;
        TextureData* dev_diffuseTex = NULL;
        glm::vec2 texcoord0;
        int texWidth;
        int texHeight;
        // ...
    };

    struct Primitive {
        PrimitiveType primitiveType = Triangle;	// C++ 11 init
        VertexOut v[3];
    };

    struct Fragment {
        glm::vec3 color;

        // TODO: add new attributes to your Fragment
        // The attributes listed below might be useful,
        // but always feel free to modify on your own

        // glm::vec3 eyePos;	// eye space position used for shading
        glm::vec3 eyeNor;
        VertexAttributeTexcoord texcoord0;
        TextureData* dev_diffuseTex = NULL;
        int diffuseTexWidth;
        int diffuseTexHeight;
        // ...
    };

    struct PrimitiveDevBufPointers {
        int primitiveMode;	//from tinygltfloader macro
        PrimitiveType primitiveType;
        int numPrimitives;
        int numIndices;
        int numVertices;

        // Vertex In, const after loaded
        VertexIndex* dev_indices;
        VertexAttributePosition* dev_position;
        VertexAttributeNormal* dev_normal;
        VertexAttributeTexcoord* dev_texcoord0;

        // Materials, add more attributes when needed
        TextureData* dev_diffuseTex;
        int diffuseTexWidth;
        int diffuseTexHeight;
        // TextureData* dev_specularTex;
        // TextureData* dev_normalTex;
        // ...

        // Vertex Out, vertex used for rasterization, this is changing every frame
        VertexOut* dev_verticesOut;
    };

}

// Pipeline Options
#define LINE_INTERSECTION_RASTERIZATION // If on, will compute line intersections with triangle edges for scan line conversion. If off, will naively check every pixel in the triangle's bounding box
#define DEPTH_TEST
#define TEXTURE_MAPPING
//#define SSAA

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;

static int width = 0;
static int height = 0;

// for mutex
static int* dev_mutex = NULL;
static int* dev_mutex_4X = NULL;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;

// All _4X versions are twice as large for the purposes of super sampling
static Fragment *dev_fragmentBuffer = NULL;
static Fragment *dev_fragmentBuffer_4X = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static glm::vec3 *dev_framebuffer_4X = NULL;

static int * dev_depth = NULL;
static int * dev_depth_4X = NULL;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__device__
glm::vec3 BilinearFilter(const glm::vec3 color00, const glm::vec3 color01, const glm::vec3 color10, const glm::vec3 color11, const glm::vec2 weights) {
    // Interpolate along the X-axis
    const glm::vec3 colorX1 = glm::mix(color00, color01, weights.x);
    const glm::vec3 colorX2 = glm::mix(color10, color11, weights.x);

    // Interpolate along the Y-axis
    return glm::mix(colorX1, colorX2, weights.y);
}

// Return the tightly packed chars as a color with its elements on the range [0, 1]
__device__
glm::vec3 SampleTexture(const TextureData *texture, const int index) {
    const int r = texture[index * 3];
    const int g = texture[index * 3 + 1];
    const int b = texture[index * 3 + 2];
    return glm::vec3(r, g, b) * 0.00392156862f; // 1 / 255
}

// Given a texture and UVs, samples the texture and returns a color using bilinear filtering
__device__
glm::vec3 GetTextureColor(const TextureData *texture, const glm::vec2 uv, const int width, const int height) {
    
    // Compute the 4 indices into the texture
    const glm::vec2 texIndex = uv * glm::vec2(width, height);

    const glm::vec2 sample00 = floor(texIndex);
    const glm::vec2 sample01 = glm::vec2(ceil(texIndex.x), floor(texIndex.y));
    const glm::vec2 sample10 = glm::vec2(floor(texIndex.x), ceil(texIndex.y));
    const glm::vec2 sample11 = ceil(texIndex);

    const glm::vec2 weights = glm::vec2((texIndex.x - sample10.x) / (sample01.x - sample10.x),
                                        (texIndex.y - sample01.y) / (sample10.y - sample01.y));
    
    // Sample the colors
    const glm::vec3 color00 = SampleTexture(texture, ((int)sample00.x) + ((int)sample00.y) * width);
    const glm::vec3 color01 = SampleTexture(texture, ((int)sample01.x) + ((int)sample01.y) * width);
    const glm::vec3 color10 = SampleTexture(texture, ((int)sample10.x) + ((int)sample10.y) * width);
    const glm::vec3 color11 = SampleTexture(texture, ((int)sample11.x) + ((int)sample11.y) * width);

    return BilinearFilter(color00, color01, color10, color11, weights);
}

// Downsample the high-res rendered image to the normal image resolution using some reconstruction filter - grid, RGSS, Quincunx, etc
__global__
void downsample(const int w, const int h, glm::vec3 *framebuffer, const glm::vec3 *framebuffer_4X) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        // Downsample using uniform grid sampling

        // Compute index in the larger resolution image
        const int index_4X = 2 * x + 4 * y * w;

        // Access colors and average
        glm::vec3 color = framebuffer_4X[index_4X];
        color += framebuffer_4X[index_4X + 1];
        color += framebuffer_4X[index_4X + w * 2];
        color += framebuffer_4X[index_4X + 1 + w * 2];

        framebuffer[index] = color * 0.25f;
    }
}

/**
* Writes fragment colors to the framebuffer
*/
__global__
void render(const int w, const int h, const Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        const Fragment &fragment = fragmentBuffer[index];

        // Lighting
        const glm::vec3 lightVec = glm::normalize(glm::vec3(1.0f, 1.0f, 0.0f));
        const float lightIntensity = glm::dot(lightVec, fragment.eyeNor) * 0.5f + 0.5f;

        #ifdef TEXTURE_MAPPING
        if (fragment.dev_diffuseTex) {
            const glm::vec3 texCol = GetTextureColor(fragment.dev_diffuseTex, fragment.texcoord0, fragment.diffuseTexWidth, fragment.diffuseTexHeight);
            framebuffer[index] = texCol *lightIntensity;
        } else {
            framebuffer[index] = fragment.color;
        }
        #else
        framebuffer[index] = fragment.color;
        #endif

    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;

    #ifndef SSAA
    // Fragment Buffer
    cudaFree(dev_fragmentBuffer);
    cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
    cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    #else
    // Fragment Buffer 4X
    cudaFree(dev_fragmentBuffer_4X);
    cudaMalloc(&dev_fragmentBuffer_4X, 4 * width * height * sizeof(Fragment));
    cudaMemset(dev_fragmentBuffer_4X, 0, 4 * width * height * sizeof(Fragment));
    #endif

    // Frame Buffer
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    #ifdef SSAA
    // Frame Buffer 4X
    cudaFree(dev_framebuffer_4X);
    cudaMalloc(&dev_framebuffer_4X, 4 * width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer_4X, 0, 4 * width * height * sizeof(glm::vec3));
    #endif

    #ifndef SSAA
    // Depth Buffer
    cudaFree(dev_depth);
    cudaMalloc(&dev_depth, width * height * sizeof(int));
    #else
    // Depth Buffer 4X
    cudaFree(dev_depth_4X);
    cudaMalloc(&dev_depth_4X, 4 * width * height * sizeof(int));
    #endif

    #ifndef SSAA
    // Mutex array
    cudaFree(dev_mutex);
    cudaMalloc(&dev_mutex, width * height * sizeof(int));
    cudaMemset(dev_mutex, 0, width * height * sizeof(int));
    #else
    // Mutex array 4X
    cudaFree(dev_mutex_4X);
    cudaMalloc(&dev_mutex_4X, 4 * width * height * sizeof(int));
    cudaMemset(dev_mutex_4X, 0, 4 * width * height * sizeof(int));
    #endif

    checkCUDAError("rasterizeInit");
}

__global__
void initDepth(const int w, const int h, int * depth)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h)
    {
        const int index = x + (y * w);
        depth[index] = INT_MAX;
    }
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {

    // Attribute (vec3 position)
    // component (3 * float)
    // byte (4 * byte)

    // id of component
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < N) {
        int count = i / n;
        int offset = i - count * n;	// which component of the attribute

        for (int j = 0; j < componentTypeByteSize; j++) {

            dev_dst[count * componentTypeByteSize * n
                + offset * componentTypeByteSize
                + j]

                =

                dev_src[byteOffset
                + count * (byteStride == 0 ? componentTypeByteSize * n : byteStride)
                + offset * componentTypeByteSize
                + j];
        }
    }


}

__global__
void _nodeMatrixTransform(
    int numVertices,
    VertexAttributePosition* position,
    VertexAttributeNormal* normal,
    glm::mat4 MV, glm::mat3 MV_normal) {

    // vertex id
    int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (vid < numVertices) {
        position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
        normal[vid] = glm::normalize(MV_normal * normal[vid]);
    }
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {

    glm::mat4 curMatrix(1.0);

    const std::vector<double> &m = n.matrix;
    if (m.size() > 0) {
        // matrix, copy it

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                curMatrix[i][j] = (float)m.at(4 * i + j);
            }
        }
    }
    else {
        // no matrix, use rotation, scale, translation

        if (n.translation.size() > 0) {
            curMatrix[3][0] = n.translation[0];
            curMatrix[3][1] = n.translation[1];
            curMatrix[3][2] = n.translation[2];
        }

        if (n.rotation.size() > 0) {
            glm::mat4 R;
            glm::quat q;
            q[0] = n.rotation[0];
            q[1] = n.rotation[1];
            q[2] = n.rotation[2];

            R = glm::mat4_cast(q);
            curMatrix = curMatrix * R;
        }

        if (n.scale.size() > 0) {
            curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
        }
    }

    return curMatrix;
}

void traverseNode(
    std::map<std::string, glm::mat4> & n2m,
    const tinygltf::Scene & scene,
    const std::string & nodeString,
    const glm::mat4 & parentMatrix
)
{
    const tinygltf::Node & n = scene.nodes.at(nodeString);
    glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
    n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

    auto it = n.children.begin();
    auto itEnd = n.children.end();

    for (; it != itEnd; ++it) {
        traverseNode(n2m, scene, *it, M);
    }
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

    totalNumPrimitives = 0;

    std::map<std::string, BufferByte*> bufferViewDevPointers;

    // 1. copy all `bufferViews` to device memory
    {
        std::map<std::string, tinygltf::BufferView>::const_iterator it(
            scene.bufferViews.begin());
        std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
            scene.bufferViews.end());

        for (; it != itEnd; it++) {
            const std::string key = it->first;
            const tinygltf::BufferView &bufferView = it->second;
            if (bufferView.target == 0) {
                continue; // Unsupported bufferView.
            }

            const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

            BufferByte* dev_bufferView;
            cudaMalloc(&dev_bufferView, bufferView.byteLength);
            cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

            checkCUDAError("Set BufferView Device Mem");

            bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

        }
    }



    // 2. for each mesh: 
    //		for each primitive: 
    //			build device buffer of indices, materail, and each attributes
    //			and store these pointers in a map
    {

        std::map<std::string, glm::mat4> nodeString2Matrix;
        auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

        {
            auto it = rootNodeNamesList.begin();
            auto itEnd = rootNodeNamesList.end();
            for (; it != itEnd; ++it) {
                traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
            }
        }


        // parse through node to access mesh

        auto itNode = nodeString2Matrix.begin();
        auto itEndNode = nodeString2Matrix.end();
        for (; itNode != itEndNode; ++itNode) {

            const tinygltf::Node & N = scene.nodes.at(itNode->first);
            const glm::mat4 & matrix = itNode->second;
            const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

            auto itMeshName = N.meshes.begin();
            auto itEndMeshName = N.meshes.end();

            for (; itMeshName != itEndMeshName; ++itMeshName) {

                const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

                auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
                std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

                // for each primitive
                for (size_t i = 0; i < mesh.primitives.size(); i++) {
                    const tinygltf::Primitive &primitive = mesh.primitives[i];

                    if (primitive.indices.empty())
                        return;

                    // TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
                    VertexIndex* dev_indices = NULL;
                    VertexAttributePosition* dev_position = NULL;
                    VertexAttributeNormal* dev_normal = NULL;
                    VertexAttributeTexcoord* dev_texcoord0 = NULL;

                    // ----------Indices-------------

                    const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
                    const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
                    BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

                    // assume type is SCALAR for indices
                    int n = 1;
                    int numIndices = indexAccessor.count;
                    int componentTypeByteSize = sizeof(VertexIndex);
                    int byteLength = numIndices * n * componentTypeByteSize;

                    dim3 numThreadsPerBlock(128);
                    dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                    cudaMalloc(&dev_indices, byteLength);
                    _deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
                        numIndices,
                        (BufferByte*)dev_indices,
                        dev_bufferView,
                        n,
                        indexAccessor.byteStride,
                        indexAccessor.byteOffset,
                        componentTypeByteSize);


                    checkCUDAError("Set Index Buffer");


                    // ---------Primitive Info-------

                    // Warning: LINE_STRIP is not supported in tinygltfloader
                    int numPrimitives;
                    PrimitiveType primitiveType;
                    switch (primitive.mode) {
                    case TINYGLTF_MODE_TRIANGLES:
                        primitiveType = PrimitiveType::Triangle;
                        numPrimitives = numIndices / 3;
                        break;
                    case TINYGLTF_MODE_TRIANGLE_STRIP:
                        primitiveType = PrimitiveType::Triangle;
                        numPrimitives = numIndices - 2;
                        break;
                    case TINYGLTF_MODE_TRIANGLE_FAN:
                        primitiveType = PrimitiveType::Triangle;
                        numPrimitives = numIndices - 2;
                        break;
                    case TINYGLTF_MODE_LINE:
                        primitiveType = PrimitiveType::Line;
                        numPrimitives = numIndices / 2;
                        break;
                    case TINYGLTF_MODE_LINE_LOOP:
                        primitiveType = PrimitiveType::Line;
                        numPrimitives = numIndices + 1;
                        break;
                    case TINYGLTF_MODE_POINTS:
                        primitiveType = PrimitiveType::Point;
                        numPrimitives = numIndices;
                        break;
                    default:
                        // output error
                        break;
                    };


                    // ----------Attributes-------------

                    auto it(primitive.attributes.begin());
                    auto itEnd(primitive.attributes.end());

                    int numVertices = 0;
                    // for each attribute
                    for (; it != itEnd; it++) {
                        const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
                        const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

                        int n = 1;
                        if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                            n = 1;
                        }
                        else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                            n = 2;
                        }
                        else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                            n = 3;
                        }
                        else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                            n = 4;
                        }

                        BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
                        BufferByte ** dev_attribute = NULL;

                        numVertices = accessor.count;
                        int componentTypeByteSize;

                        // Note: since the type of our attribute array (dev_position) is static (float32)
                        // We assume the glTF model attribute type are 5126(FLOAT) here

                        if (it->first.compare("POSITION") == 0) {
                            componentTypeByteSize = sizeof(VertexAttributePosition) / n;
                            dev_attribute = (BufferByte**)&dev_position;
                        }
                        else if (it->first.compare("NORMAL") == 0) {
                            componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
                            dev_attribute = (BufferByte**)&dev_normal;
                        }
                        else if (it->first.compare("TEXCOORD_0") == 0) {
                            componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
                            dev_attribute = (BufferByte**)&dev_texcoord0;
                        }

                        std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

                        dim3 numThreadsPerBlock(128);
                        dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                        int byteLength = numVertices * n * componentTypeByteSize;
                        cudaMalloc(dev_attribute, byteLength);

                        _deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
                            n * numVertices,
                            *dev_attribute,
                            dev_bufferView,
                            n,
                            accessor.byteStride,
                            accessor.byteOffset,
                            componentTypeByteSize);

                        std::string msg = "Set Attribute Buffer: " + it->first;
                        checkCUDAError(msg.c_str());
                    }

                    // malloc for VertexOut
                    VertexOut* dev_vertexOut;
                    cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
                    checkCUDAError("Malloc VertexOut Buffer");

                    // ----------Materials-------------

                    // You can only worry about this part once you started to 
                    // implement textures for your rasterizer
                    TextureData* dev_diffuseTex = NULL;
                    int diffuseTexWidth = 0;
                    int diffuseTexHeight = 0;
                    if (!primitive.material.empty()) {
                        const tinygltf::Material &mat = scene.materials.at(primitive.material);
                        printf("material.name = %s\n", mat.name.c_str());

                        if (mat.values.find("diffuse") != mat.values.end()) {
                            std::string diffuseTexName = mat.values.at("diffuse").string_value;
                            if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
                                const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
                                if (scene.images.find(tex.source) != scene.images.end()) {
                                    const tinygltf::Image &image = scene.images.at(tex.source);

                                    size_t s = image.image.size() * sizeof(TextureData);
                                    cudaMalloc(&dev_diffuseTex, s);
                                    cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

                                    diffuseTexWidth = image.width;
                                    diffuseTexHeight = image.height;

                                    checkCUDAError("Set Texture Image data");
                                }
                            }
                        }

                        // TODO: write your code for other materails
                        // You may have to take a look at tinygltfloader
                        // You can also use the above code loading diffuse material as a start point 
                    }


                    // ---------Node hierarchy transform--------
                    cudaDeviceSynchronize();

                    dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                    _nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
                        numVertices,
                        dev_position,
                        dev_normal,
                        matrix,
                        matrixNormal);

                    checkCUDAError("Node hierarchy transformation");

                    // at the end of the for loop of primitive
                    // push dev pointers to map
                    primitiveVector.push_back(PrimitiveDevBufPointers{
                        primitive.mode,
                        primitiveType,
                        numPrimitives,
                        numIndices,
                        numVertices,

                        dev_indices,
                        dev_position,
                        dev_normal,
                        dev_texcoord0,

                        dev_diffuseTex,
                        diffuseTexWidth,
                        diffuseTexHeight,

                        dev_vertexOut	//VertexOut
                    });

                    totalNumPrimitives += numPrimitives;

                } // for each primitive

            } // for each mesh

        } // for each node

    }


    // 3. Malloc for dev_primitives
    {
        cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
    }


    // Finally, cudaFree raw dev_bufferViews
    {

        std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
        std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());

        //bufferViewDevPointers

        for (; it != itEnd; it++) {
            cudaFree(it->second);
        }

        checkCUDAError("Free BufferView Device Mem");
    }


}

__global__
void _vertexTransformAndAssembly(
    const int numVertices,
    PrimitiveDevBufPointers primitive,
    const glm::mat4 MVP, const glm::mat4 MV, const glm::mat3 MV_normal,
    const int width, const int height) {

    // vertex id
    int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (vid < numVertices) {

        // Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
        // Then divide the pos by its w element to transform into NDC space
        // Finally transform x and y to viewport space

        // Create reference to the out vertex
        VertexOut &vOut = primitive.dev_verticesOut[vid];
        vOut.pos = MVP * glm::vec4(primitive.dev_position[vid], 1);

        // Perspective divide
        vOut.pos /= vOut.pos.w;

        // Viewport transformation
        vOut.pos.x = (vOut.pos.x * 0.5f + 0.5f) * ((float)width);
        vOut.pos.y = (1.0f - (vOut.pos.y * 0.5f + 0.5f)) * ((float)height);

        // Eye space attributes
        vOut.eyePos = glm::vec3(MV * glm::vec4(primitive.dev_position[vid], 1));
        vOut.eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);

        #ifdef TEXTURE_MAPPING
        vOut.dev_diffuseTex = primitive.dev_diffuseTex;
        vOut.texcoord0 = primitive.dev_texcoord0[vid];
        vOut.texWidth = primitive.diffuseTexWidth;
        vOut.texHeight = primitive.diffuseTexHeight;
        #else
        int colorChannel = vid % 3;
        vOut.col = glm::vec3(colorChannel == 0, colorChannel == 1, colorChannel == 2);
        #endif
    }
}

static int curPrimitiveBeginId = 0;

__global__
void _primitiveAssembly(const int numIndices, const int curPrimitiveBeginId, Primitive* dev_primitives, const PrimitiveDevBufPointers primitive) {

    // index id
    int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iid < numIndices) {
        // This is primitive assembly for triangles
        int pid;	// id for cur primitives vector
        if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
            pid = iid / (int)primitive.primitiveType;
            dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
                = primitive.dev_verticesOut[primitive.dev_indices[iid]];
        }
    }

}

__global__
void rasterize(const int nPrims, const Primitive *primitives, Fragment *fragmentBuffer, int *depthBuffer, int* mutex_array, const int width, const int height) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < nPrims) {
        // Need an array containing the triangle vertices for helper functions in rasterizeTools.h.
        const auto &primVerts = primitives[tid].v;
        const glm::vec3 triVerts[] = { glm::vec3(primVerts[0].pos), glm::vec3(primVerts[1].pos), glm::vec3(primVerts[2].pos) };

        // Compute the bouding box for this primitive
        const AABB aabb = getAABBForTriangle(triVerts);

        // Transform the aabb's min/max x and y values into pixel coordinates
        const int aabbMin[] = { (int)floor(aabb.min.x),
                                glm::clamp((int)floor(aabb.min.y), 0, height - 1) };
        const int aabbMax[] = { (int)ceil(aabb.max.x),
                                glm::clamp((int)ceil(aabb.max.y), 0, height - 1) };

        for (int y = aabbMin[1]; y <= aabbMax[1]; ++y) {

            float xIsectMin, xIsectMax;
            #ifdef LINE_INTERSECTION_RASTERIZATION

            // Compute intersection with each line segment pair
            xIsectMin = 1000000000.0f;
            xIsectMax = -1000000000.0f;
            for (int l = 0; l < 3; ++l) {
                // Edge cases
                if (triVerts[(l + 1) % 3].y == triVerts[l].y) continue; // horizontal line
                if (triVerts[(l + 1) % 3].x == triVerts[l].x) { // vertical line
                    xIsectMin = glm::min(xIsectMin, triVerts[l].x);
                    xIsectMax = glm::max(xIsectMax, triVerts[l].x);
                    continue;
                }

                // Compute intersection with line
                const float slope = (triVerts[(l + 1) % 3].y - triVerts[l].y) / ((triVerts[(l + 1) % 3].x - triVerts[l].x));
                float xIsect = (y - triVerts[l].y + slope * triVerts[l].x) / slope;

                // Ignore intersections outside of bounding boxes
                if (xIsect < aabbMin[0] || xIsect > aabbMax[0]) continue;

                xIsectMin = glm::min(xIsectMin, xIsect);
                xIsectMax = glm::max(xIsectMax, xIsect);
            }
            #else
            xIsectMin = aabbMin[0];
            xIsectMax = aabbMax[0];
            #endif

            xIsectMin = glm::clamp(xIsectMin, 0.0f, (float)width - 1);
            xIsectMax = glm::clamp(xIsectMax, 0.0f, (float)width - 1);

            // Set interpolated values for all overlapping fragments
            for (int x = (int)xIsectMin; x <= (int)xIsectMax; ++x) {
                const int fragIndex = x + y * width;
                const glm::vec3 baryWeights = calculateBarycentricCoordinate(triVerts, glm::vec2(x, y));
                if (isBarycentricCoordInBounds(baryWeights)) {
                    // Compute the z value using perspective correct interpolation. Note the depth buffer expects ints
                    const int zDepth = (int)((10000000.0f) / (baryWeights.x / triVerts[0].z + baryWeights.y / triVerts[1].z + baryWeights.z / triVerts[2].z));

                    // Using a mutex + atomic functions to avoid race conditions, compare with and potentially write to the depth buffer
                    bool shouldSetFragment = true;

                    #ifdef DEPTH_TEST
                    shouldSetFragment = false;
                    int *mutex = mutex_array + fragIndex; // pointer arithmetic
                                                              // Loop- wait until this thread is able to execute its critical section.
                    bool isSet = false;
                    do {
                        isSet = (atomicCAS(mutex, 0, 1) == 0);
                        if (isSet) {
                            if (zDepth < depthBuffer[fragIndex]) {
                                depthBuffer[fragIndex] = zDepth;
                                shouldSetFragment = true;
                            }
                            *mutex = 0;
                        }
                    } while (!isSet);
                    #endif
                    // Compute the perspective-correct-ly-interpolated fragment parameters
                    if (shouldSetFragment) {
                        #ifdef TEXTURE_MAPPING
                        // Texture Data - assume the texture data is the same for each vertex
                        fragmentBuffer[fragIndex].dev_diffuseTex = primVerts[0].dev_diffuseTex;
                        fragmentBuffer[fragIndex].diffuseTexWidth = primVerts[0].texWidth;
                        fragmentBuffer[fragIndex].diffuseTexHeight = primVerts[0].texHeight;

                        // UVs
                        const float zDepth_f = ((float)zDepth) / 10000000.0f;
                        fragmentBuffer[fragIndex].texcoord0 = zDepth_f * (primVerts[0].texcoord0 * baryWeights.x / triVerts[0].z +
                                                                          primVerts[1].texcoord0 * baryWeights.y / triVerts[1].z +
                                                                          primVerts[2].texcoord0 * baryWeights.z / triVerts[2].z);

                        /*fragmentBuffer[fragIndex].texcoord0 = primVerts[0].texcoord0 * baryWeights.x +
                                                              primVerts[1].texcoord0 * baryWeights.y +
                                                              primVerts[2].texcoord0 * baryWeights.z;*/
                        #else
                        fragmentBuffer[fragIndex].color = glm::abs(zDepth_f * ((primVerts[0].col * baryWeights.x / triVerts[0].z +
                                                                                primVerts[1].col * baryWeights.y / triVerts[1].z +
                                                                                primVerts[2].col * baryWeights.z / triVerts[2].z)));
                        #endif

                        // Normal
                        fragmentBuffer[fragIndex].eyeNor = glm::normalize(zDepth_f * ((primVerts[0].eyeNor * baryWeights.x / triVerts[0].z +
                                                                                       primVerts[1].eyeNor * baryWeights.y / triVerts[1].z +
                                                                                       primVerts[2].eyeNor * baryWeights.z / triVerts[2].z)));

                        /*fragmentBuffer[fragIndex].eyeNor = glm::normalize(primVerts[0].eyeNor * baryWeights.x +
                        primVerts[1].eyeNor * baryWeights.y +
                        primVerts[2].eyeNor * baryWeights.z);*/
                    }
                }
            }
        }
    }
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    const int sideLength2d = 8;
    const dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d;

    #ifdef SSAA
    blockCount2d = dim3((width * 2 - 1) / blockSize2d.x + 1,
                        (height * 2 - 1) / blockSize2d.y + 1);
    #else
    blockCount2d = dim3((width - 1) / blockSize2d.x + 1,
                        (height - 1) / blockSize2d.y + 1);
    #endif

    // Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

    // Vertex Process & primitive assembly
    {
        curPrimitiveBeginId = 0;
        const dim3 numThreadsPerBlock(128);

        auto it = mesh2PrimitivesMap.begin();
        auto itEnd = mesh2PrimitivesMap.end();

        for (; it != itEnd; ++it) {
            auto p = (it->second).begin();	// each primitive
            auto pEnd = (it->second).end();
            for (; p != pEnd; ++p) {
                dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
                dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

                #ifdef SSAA
                _vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> > (p->numVertices, *p, MVP, MV, MV_normal, width * 2, height * 2);
                #else
                _vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> > (p->numVertices, *p, MVP, MV, MV_normal, width, height);
                #endif
                checkCUDAError("Vertex Processing");
                cudaDeviceSynchronize();

                _primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
                    (p->numIndices,
                        curPrimitiveBeginId,
                        dev_primitives,
                        *p);
                checkCUDAError("Primitive Assembly");

                curPrimitiveBeginId += p->numPrimitives; 
            }
        }

        checkCUDAError("Vertex Processing and Primitive Assembly");
    }

    // Parameters for rasterization kernel
    const int blockSize1d(totalNumPrimitives);
    const dim3 blockCount1d((totalNumPrimitives + blockSize1d - 1) / blockSize1d);

    #ifdef SSAA
    cudaMemset(dev_fragmentBuffer_4X, 0, 4 * width * height * sizeof(Fragment));
    initDepth << <blockCount2d, blockSize2d >> > (width * 2, height * 2, dev_depth_4X);
    
    // Compute attributes for fragments overlapping geometry
    rasterize << <blockSize1d, blockCount1d >> > (totalNumPrimitives, dev_primitives, dev_fragmentBuffer_4X, dev_depth_4X, dev_mutex_4X, width * 2, height * 2);
    cudaDeviceSynchronize();
    checkCUDAError("Rasterization");

    // Copy depthbuffer colors into framebuffer
    render << <blockCount2d, blockSize2d >> > (width * 2, height * 2, dev_fragmentBuffer_4X, dev_framebuffer_4X);
    cudaDeviceSynchronize();
    checkCUDAError("fragment shader");

    // Downsample the rendered image to the final image size using bilinear filtering
    downsample << <blockCount2d, blockSize2d >> > (width, height, dev_framebuffer, dev_framebuffer_4X);
    checkCUDAError("downsample SSAA");

    #else
    cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    initDepth << <blockCount2d, blockSize2d >> > (width, height, dev_depth);

    rasterize << <blockSize1d, blockCount1d >> > (totalNumPrimitives, dev_primitives, dev_fragmentBuffer, dev_depth, dev_mutex, width, height);
    cudaDeviceSynchronize();
    checkCUDAError("Rasterization");

    render << <blockCount2d, blockSize2d >> > (width, height, dev_fragmentBuffer, dev_framebuffer);
    cudaDeviceSynchronize();
    checkCUDAError("fragment shader");
    #endif

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO << <blockCount2d, blockSize2d >> > (pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

    auto it(mesh2PrimitivesMap.begin());
    auto itEnd(mesh2PrimitivesMap.end());
    for (; it != itEnd; ++it) {
        for (auto p = it->second.begin(); p != it->second.end(); ++p) {
            cudaFree(p->dev_indices);
            cudaFree(p->dev_position);
            cudaFree(p->dev_normal);
            cudaFree(p->dev_texcoord0);
            cudaFree(p->dev_diffuseTex);

            cudaFree(p->dev_verticesOut);


            //TODO: release other attributes and materials
        }
    }

    ////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

    cudaFree(dev_fragmentBuffer);
    dev_fragmentBuffer = NULL;
    cudaFree(dev_fragmentBuffer_4X);
    dev_fragmentBuffer_4X = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;
    cudaFree(dev_framebuffer_4X);
    dev_framebuffer_4X = NULL;

    cudaFree(dev_mutex);
    dev_mutex = NULL;
    cudaFree(dev_mutex_4X);
    dev_mutex_4X = NULL;

    cudaFree(dev_depth);
    dev_depth = NULL;
    cudaFree(dev_depth_4X);
    dev_depth_4X = NULL;

    checkCUDAError("rasterize Free");
}
