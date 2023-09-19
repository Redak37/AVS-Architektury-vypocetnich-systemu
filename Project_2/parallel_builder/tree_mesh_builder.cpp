/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Radek Duchoò <xducho07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    18. 12. 2020
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::evaluateCube(const ParametricScalarField &field, const unsigned gridSize, const Vec3_t<float> offset)
{
    unsigned totalTriangles = 0;
    // gridSize is already halved, therefore it is compared to 0
    if (gridSize)
    {
        const Vec3_t<float> midPoint{ (offset.x + gridSize) * mGridResolution,
            (offset.y + gridSize) * mGridResolution,
            (offset.z + gridSize) * mGridResolution };

        // control fo empty cube, gridSize is already halved, division from sqrt(3)/2 is hidden there
        if (evaluateFieldAt(midPoint, field) > mIsoLevel + sqrt(3) * gridSize * mGridResolution)
            return 0;

        const unsigned gridX = offset.x + gridSize;
        const unsigned gridY = offset.y + gridSize;
        const unsigned gridZ = offset.z + gridSize;

        // create 8 tasks
        #pragma omp task default(none) shared(field, totalTriangles)
        #pragma omp atomic update
        totalTriangles += evaluateCube(field, gridSize >> 1,
            Vec3_t<float>{offset.x, offset.y, offset.z});

        #pragma omp task default(none) shared(field, totalTriangles)
        #pragma omp atomic update
        totalTriangles += evaluateCube(field, gridSize >> 1,
            Vec3_t<float>{offset.x, offset.y, gridZ});

        #pragma omp task default(none) shared(field, totalTriangles)
        #pragma omp atomic update
        totalTriangles += evaluateCube(field, gridSize >> 1,
            Vec3_t<float>{offset.x, gridY, offset.z});

        #pragma omp task default(none) shared(field, totalTriangles)
        #pragma omp atomic update
        totalTriangles += evaluateCube(field, gridSize >> 1,
            Vec3_t<float>{offset.x, gridY, gridZ});

        #pragma omp task default(none) shared(field, totalTriangles)
        #pragma omp atomic update
        totalTriangles += evaluateCube(field, gridSize >> 1,
            Vec3_t<float>{gridX, offset.y, offset.z});

        #pragma omp task default(none) shared(field, totalTriangles)
        #pragma omp atomic update
        totalTriangles += evaluateCube(field, gridSize >> 1,
            Vec3_t<float>{gridX, offset.y, gridZ});

        #pragma omp task default(none) shared(field, totalTriangles)
        #pragma omp atomic update
        totalTriangles += evaluateCube(field, gridSize >> 1,
            Vec3_t<float>{gridX, gridY, offset.z});

        #pragma omp task default(none) shared(field, totalTriangles)
        #pragma omp atomic update
        totalTriangles += evaluateCube(field, gridSize >> 1,
            Vec3_t<float>{gridX, gridY, gridZ});

    } else {
        return buildCube(offset, field);
    }


    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned totalTriangles = 0;

    #pragma omp parallel default(none) shared(field, totalTriangles)
    #pragma omp single nowait
    totalTriangles = evaluateCube(field, mGridSize >> 1, Vec3_t<float>{0, 0, 0});

    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    static const Vec3_t<float>* const pPoints = field.getPoints().data();
    static const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    //#pragma omp parallel for default(none) shared(pos, pPoints) reduction(min: value) schedule(static)
    for (unsigned i = 0; i < count; ++i)
    {
        // (A - B)^2 == A^2 - 2 * A * B + B^2; A^2 can be substituted as it will be always same
        const float distanceSquared = pPoints[i].x * pPoints[i].x
            + pPoints[i].y * pPoints[i].y
            + pPoints[i].z * pPoints[i].z
            - 2 * (pos.x * pPoints[i].x + pos.y * pPoints[i].y + pos.z * pPoints[i].z);

        // Comparing 2 * A * B + B ^2 insted of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance with A^2 added to get the real distance
    return sqrt(value + pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
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
