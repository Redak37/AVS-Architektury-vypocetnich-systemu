/**
 * @file    cached_mesh_builder.h
 *
 * @author  Radek Duchoò <xducho07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using pre-computed field
 *
 * @date    19. 12. 2020
 **/

#ifndef CACHED_MESH_BUILDER_H
#define CACHED_MESH_BUILDER_H

#include <vector>
#include "base_mesh_builder.h"

class CachedMeshBuilder : public BaseMeshBuilder
{
public:
    CachedMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }

    std::vector<Triangle_t> mTriangles; ///< Temporary array of triangles
    float *cache; // saved pre-calculated data for evaluateFieldAt
    float invMGridResolution; // == 1 / mGridResolution (to avoid multiple division)
};

#endif // CACHED_MESH_BUILDER_H
