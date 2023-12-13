module ImageComponentAnalysis


using AbstractTrees
# Used in generic_labelling.jl to allow @nexprs macros.
using Base.Cartesian
using DataFrames
using ImageFiltering: padarray, Fill
using LeftChildRightSiblingTrees
using LinearAlgebra
using OffsetArrays: OffsetVector
using Parameters
using ConvexHulls2d
using StaticArrays


# TODO: port ComponentAnalysisAPI to ImagesAPI
include("ComponentAnalysisAPI/ComponentAnalysisAPI.jl")
import .ComponentAnalysisAPI: AbstractComponentAnalysisAlgorithm,
                              analyze_components, analyze_components!

include("algorithms/bitcodes.jl")
include("algorithms/basic_measurement.jl")
include("algorithms/basic_topology.jl")
include("algorithms/bounding_box.jl")
include("algorithms/contour_topology.jl")
include("algorithms/ellipse_region.jl")
include("algorithms/minimum_oriented_bounding_box.jl")
include("label_components.jl")


export
    # Types of analysis one can perform on a labelled binary image.
    analyze_components,
    analyze_components!,
    BasicMeasurement,
    BasicTopology,
    BoundingBox,
    Contour,
    EllipseRegion,
    #ContourTopology, TODO
    MinimumOrientedBoundingBox,
    establish_contour_hierarchy,
    label_components
end # module
