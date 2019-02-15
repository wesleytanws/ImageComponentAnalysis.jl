"""
```
label_components(IterativeRecursionOriginal3(), binary_volume, local_window)
```

Computes the connected components of a binary volume using 26-connectivity.

# Output

Returns a three-dimensional `Int` array that represents the label of every voxel.

# Arguments

## `binary_volume::AbstractArray`

The binary volume made up of values 0 (black) and 1 (white).

## `local_window::NTuple{3, Int}`

The size of the local window for each iteration.

# Example

Computes the labels for a randomly generated 20 x 20 x 20 array using 5 x 7 x 7 local window.

```julia
using ImageComponentAnalysis

binary_volume = Int.(rand(Bool, 20, 20, 20))

labels = label_components(IterativeRecursionOriginal3(), binary_volume, (5, 7, 7))
```

# Reference

[1] Q. Hu, G. Qian, and W.L. Nowinski, “Fast connected-component labelling in three-dimensional binary images based on iterative recursion”, Computer Vision and Image Understanding, vol. 99, no. 3, pp. 414–434, Sep. 2005. [doi:10.1016/j.cviu.2005.04.001](https://doi.org/10.1016/j.cviu.2005.04.001)
"""
function label_components(algorithm::IterativeRecursionOriginal3, binary_volume::AbstractArray, local_window::NTuple{3, Int})

    @inbounds begin
        height, width, length = size(binary_volume)
        queue = Queue{NTuple{3, Int}}()

        n₁, n₂, n₃ = map((b, l) -> ceil(Int, (max(1, min(b, l)-1)/2)), size(binary_volume), local_window)
        labels = Int.(copy(binary_volume))
        labelindex = 3
        subvolume = Array{Int}(undef, map(n -> 2n+1, (n₁, n₂, n₃)))

        for x = 1:height, y = 1:width, z = 1:length
            if (labels[x, y, z] == 1)
                enqueue!(queue, (x, y, z))
                while !isempty(queue)
                    vᵢx, vᵢy, vᵢz = dequeue!(queue)
                    formulate_subvolume!(vᵢx, vᵢy, vᵢz, n₁, n₂, n₃, height, width, length, labels, subvolume)
                    label_subvolume!(algorithm, vᵢx, vᵢy, vᵢz, n₁, n₂, n₃, queue, labelindex, subvolume)
                    apply_labels!(vᵢx, vᵢy, vᵢz, n₁, n₂, n₃, labels, labelindex, subvolume)
                end
                labelindex += 1
            end
        end

        replace!(s -> s >= 3 ? s-2 : 0, labels)
    end
end

# Formulate subvolume of size (2n₁+1) x (2n₂+1) x (2n₃+1) from the labels.
function formulate_subvolume!(vᵢx, vᵢy, vᵢz, n₁, n₂, n₃, height, width, length, labels, subvolume)

    initial = map((x, n) -> max(x-n, 1), (vᵢx, vᵢy, vᵢz), (n₁, n₂, n₃))
    final = map((x, n, m) -> min(x+n, m), (vᵢx, vᵢy, vᵢz), (n₁, n₂, n₃), (height, width, length))

    range = map((u, v) -> u:v, initial, final)
    displace = map((x, n, u, v) -> n+1-(x-u):n+1-(x-v), (vᵢx, vᵢy, vᵢz), (n₁, n₂, n₃), initial, final)

    fill!(subvolume, 0)
    copyto!(subvolume, CartesianIndices(displace), labels, CartesianIndices(range))
end

# Label 26-connected voxels in the subvolume with the labelindex.
function label_subvolume!(algorithm::IterativeRecursionOriginal3, vᵢx, vᵢy, vᵢz, n₁, n₂, n₃, queue, labelindex, subvolume)

    window_size = 0
    center = (n₁ + 1, n₂ + 1, n₃ + 1)

    # Set center voxel of subvolume as labelindex.
    subvolume[n₁ + 1, n₂ + 1, n₃ + 1] = labelindex

    # Label 3 x 3 x 3 neighbourhood of center voxel.
    for (x, y, z) = get_neighbours(algorithm, 1, center, (n₁, n₂, n₃))
        if (subvolume[x, y, z] == 1)
            subvolume[x, y, z] = -10
        end
    end

    # Grow window size to local window.
    while window_size != max(n₁, n₂, n₃) + 1
        for (vx, vy, vz) = get_neighbours(algorithm, window_size, center, (n₁, n₂, n₃))

            # Label object voxels with distance of window_size.
            if (subvolume[vx, vy, vz] == -10)
                if (abs(vx-(n₁ + 1)) == n₁ || abs(vy-(n₂ + 1)) == n₂ || abs(vz-(n₃ + 1)) == n₃)
                    enqueue!(queue, (vᵢx + vx - (n₁ + 1), vᵢy + vy - (n₂ + 1), vᵢz + vz - (n₃ + 1)))
                end

                # Label 3 x 3 x 3 neighbourhood of current object voxel.
                for (x, y, z) = get_neighbours(algorithm, 1, (vx, vy, vz), (n₁, n₂, n₃))
                    if (subvolume[x, y, z] == 1)
                        subvolume[x, y, z] = -10
                        if (abs(x-(n₁ + 1)) == n₁ || abs(y-(n₂ + 1)) == n₂ || abs(z-(n₃ + 1)) == n₃)
                            enqueue!(queue, (vᵢx + x -(n₁ + 1), vᵢy + y - (n₂ + 1), vᵢz + z - (n₃ + 1)))
                        end
                    end
                end

                subvolume[vx, vy, vz] = labelindex
            end
        end
        window_size += 1
    end

    # Replace and queue all object voxels with label -10.
    for i = Iterators.filter(i -> subvolume[i] == -10, CartesianIndices(subvolume))
        enqueue!(queue, (vᵢx + i[1] - (n₁ + 1), vᵢy + i[2] - (n₂ + 1), vᵢz + i[3] - (n₃ + 1)))
        subvolume[i] = labelindex
    end
end

# Get the neighbourhood of the center voxel.
function get_neighbours(algorithm::IterativeRecursionOriginal3, distance, center, n)

    neighbour_ranges = map((c, n) -> max(1,c-distance):min(2n+1,c+distance), center, n)

    function select_neighbours((x, y, z))
        max(abs(x-center[1]), abs(y-center[2]), abs(z-center[3])) == distance
    end

    Iterators.filter(select_neighbours, Iterators.product(neighbour_ranges...))
end

# Apply labels with the newly labelled voxels in the subvolume.
function apply_labels!(vᵢx, vᵢy, vᵢz, n₁, n₂, n₃, labels, labelindex, subvolume)
    for s in CartesianIndices(subvolume)
        if subvolume[s] == labelindex
            labels[vᵢx + s[1] - (n₁ + 1), vᵢy + s[2] - (n₂ + 1), vᵢz + s[3] - (n₃ + 1)] = labelindex
        end
    end
end
