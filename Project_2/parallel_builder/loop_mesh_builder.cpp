/**
 * @file    loop_mesh_builder.cpp
 *
 * @author  Radek Duchoò <xducho07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 *
 * @date    17. 12. 2020
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "loop_mesh_builder.h"

LoopMeshBuilder::LoopMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "OpenMP Loop")
{

}

unsigned LoopMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // 1. Compute total number of cubes in the grid.
    const size_t totalCubesCount = mGridSize * mGridSize * mGridSize;

    unsigned totalTriangles = 0;

    // 2. Loop over each coordinate in the 3D grid.
    #pragma omp parallel for default(none) shared(field) reduction(+: totalTriangles) schedule(dynamic,64)
    for (size_t i = 0; i < totalCubesCount; ++i)
    {
        // 3. Compute 3D position in the grid.
        Vec3_t<float> cubeOffset(i % mGridSize,
            (i / mGridSize) % mGridSize,
            i / (mGridSize * mGridSize));

        // 4. Evaluate "Marching Cube" at given position in the grid and
        //    store the number of triangles generated.
        totalTriangles += buildCube(cubeOffset, field);
    }

    // 5. Return total number of triangles generated.
    return totalTriangles;
}

float LoopMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    static const Vec3_t<float>* const pPoints = field.getPoints().data();
    static const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    //#pragma omp parallel for default(none) shared(pos) reduction(min: value) schedule(static)
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

void LoopMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
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
