/**
 * @file    cached_mesh_builder.cpp
 *
 * @author  Radek Duchoò <xducho07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using pre-computed field
 *
 * @date    19. 12. 2020
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "cached_mesh_builder.h"

CachedMeshBuilder::CachedMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Cached")
{

}

unsigned CachedMeshBuilder::marchCubes(const ParametricScalarField &field)
{

    invMGridResolution = 1. / mGridResolution;

    // 1. pre-calculation
    {
        const size_t preCubesCount = (mGridSize + 1) * (mGridSize + 1) * (mGridSize + 1);
        cache = new float[preCubesCount];

        const Vec3_t<float>* const pPoints = field.getPoints().data();
        const unsigned count = unsigned(field.getPoints().size());

        #pragma omp parallel for default(none) schedule(guided)
        for (size_t i = 0; i < preCubesCount; ++i)
        {
            float value = std::numeric_limits<float>::max();
            const float x = (i % (mGridSize + 1)) * mGridResolution;
            const float y = ((i / (mGridSize + 1)) % (mGridSize + 1)) * mGridResolution;
            const float z = (i / ((mGridSize + 1) * (mGridSize + 1))) * mGridResolution;

            for (unsigned j = 0; j < count; ++j)
            {
                // (A - B)^2 == A^2 - 2 * A * B + B^2; A^2 can be substituted as it will be always same
                const float distanceSquared = pPoints[j].x * pPoints[j].x
                    + pPoints[j].y * pPoints[j].y
                    + pPoints[j].z * pPoints[j].z
                    - 2 * (x * pPoints[j].x + y * pPoints[j].y + z * pPoints[j].z);

                // Comparing 2 * A * B + B ^2 insted of real distance to avoid unnecessary
                // "sqrt"s in the loop.
                value = std::min(value, distanceSquared);
            }

            // Finally take square root of the minimal square distance with A^2 added to get the real distance
            cache[i] = sqrt(value + x * x + y * y + z * z);
        }
    }


    //const size_t totalCubesCount = mGridSize * mGridSize * mGridSize;
    unsigned totalTriangles = 0;

    // 2. Loop over each coordinate in the 3D grid.
    #pragma omp parallel for default(none) shared(field) reduction(+: totalTriangles) schedule(static)
    for (size_t x = 0; x < mGridSize; ++x)
    {
        for (size_t y = 0; y < mGridSize; ++y)
        {
            for (size_t z = 0; z < mGridSize; ++z)
            {
                totalTriangles += buildCube(Vec3_t<float>{x, y, z}, field);
            }
        }
    }

    // 3. Return total number of triangles generated.
    return totalTriangles;
}

float CachedMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    return cache[(size_t)(pos.x * invMGridResolution + 0.5)
        + (size_t)(pos.y * invMGridResolution + 0.5) * ((mGridSize + 1))
        + (size_t)(pos.z * invMGridResolution + 0.5) * ((mGridSize + 1)) * ((mGridSize + 1))];
}

void CachedMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    #pragma omp critical(push_triangle)
    {
        mTriangles.push_back(triangle);
    }
}