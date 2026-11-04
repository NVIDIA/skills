# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0


# Softmax — cuTile.jl
#
# Three strategies (forward only):
#   1. TMA single-tile:  ct.load/ct.store, persistent scheduling, TILE_SIZE >= N
#   2. Online 2-pass:    ct.load/ct.store, running max + sum, one block per vector
#   3. Chunked 3-pass:   ct.gather/ct.scatter, explicit max → sum → normalize
#
# Matches julia/kernels/softmax.jl
#
# Layout: the Python version takes a row-major (M, N) tensor and reduces each
# row (last axis, contiguous). The Julia version takes the SAME memory as a
# column-major (N, M) matrix and reduces each column (dims=1, contiguous).
# Do not keep Python's (M, N) shape and reduce over dims=2: in column-major
# that axis has stride M, and the kernels run 3-8x slower on B200.
#
# Key translation patterns demonstrated:
#   - Reverse the dims of row-major tensors so the reduced axis stays contiguous
#   - Broadcast dot syntax: exp.(), .-, ./, max.()
#   - ct.PaddingMode.NegInf  (Python: ct.PaddingMode.NEG_INF)
#   - maximum(tile; dims=1)   (Python: ct.max(tile, 1, keepdims=True) on the transposed shape)
#   - fill(-Inf32, (1, 1)) for scalar accumulators
#   - zeros(Float32, 1, 1) for zero-initialized accumulators
#   - ct.Constant() at launch, plain ::Int in kernel signature
#   - Host functions accept CuMatrix{T} directly and launch asynchronously

using CUDA
using cuTile: cuTile
import cuTile as ct

#=============================================================================
 Strategy 1: TMA Single-Tile  (TILE_SIZE >= N)
 Loads an entire column in one ct.load with NegInf padding.
 Uses persistent scheduling: each block processes multiple columns.
=============================================================================#
function softmax_kernel_tma(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                            TILE_SIZE::Int) where {T}
    pid = ct.bid(1)
    num_programs = ct.num_blocks(1)
    n_cols = size(input, 2)

    col_idx = pid
    while col_idx <= n_cols
        col = ct.load(input; index=(Int32(1), col_idx), shape=(TILE_SIZE, 1),
                      padding_mode=ct.PaddingMode.NegInf)
        col = convert(ct.Tile{Float32}, col)

        col_max = maximum(col; dims=1)
        numerator = exp.(col .- col_max)
        denominator = sum(numerator; dims=1)
        softmax_output = numerator ./ denominator

        ct.store(output; index=(Int32(1), col_idx),
                 tile=convert(ct.Tile{T}, softmax_output))
        col_idx += num_programs
    end
    return
end

#=============================================================================
 Strategy 2: Online 2-Pass  (large N, one block per column)
 Pass 1: running max + sum via online algorithm (m_prev, l_prev)
 Pass 2: normalize each tile chunk
=============================================================================#
function softmax_kernel_online(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                               TILE_SIZE::Int) where {T}
    ct.@compiler_options occupancy=8

    col_idx = ct.bid(1)
    num_row_tiles = ct.num_tiles(input, 1, (TILE_SIZE, 1))

    m_prev = fill(-Inf32, (1, 1))
    l_prev = zeros(Float32, 1, 1)

    # Pass 1: compute running max and sum
    for row_idx in Int32(1):num_row_tiles
        tile = ct.load(input; index=(row_idx, col_idx), shape=(TILE_SIZE, 1),
                       padding_mode=ct.PaddingMode.NegInf)
        tile = convert(ct.Tile{Float32}, tile)

        tile_max = maximum(tile; dims=1)
        m_curr = max.(tile_max, m_prev)

        # Correct old sum: l_prev *= exp(m_prev - m_curr)
        l_prev = l_prev .* exp.(m_prev .- m_curr)

        # Update with current tile
        p = exp.(tile .- m_curr)
        l_prev = sum(p; dims=1) .+ l_prev
        m_prev = m_curr
    end

    # Pass 2: compute actual softmax values
    for row_idx in Int32(1):num_row_tiles
        tile = ct.load(input; index=(row_idx, col_idx), shape=(TILE_SIZE, 1),
                       padding_mode=ct.PaddingMode.NegInf)
        tile = convert(ct.Tile{Float32}, tile)

        numerator = exp.(tile .- m_prev)
        softmax_output = numerator ./ l_prev

        ct.store(output; index=(row_idx, col_idx),
                 tile=convert(ct.Tile{T}, softmax_output))
    end
    return
end

#=============================================================================
 Strategy 3: Chunked 3-Pass  (one block per column, gather/scatter)
 Pass 1: column max across all chunks
 Pass 2: sum of exp(x - max)
 Pass 3: normalize and scatter back
=============================================================================#
function softmax_kernel_chunked(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                                n_rows::Int, TILE_SIZE::Int) where {T}
    ct.@compiler_options occupancy=4

    col_idx = ct.bid(1)
    # Narrow to Int32 so the chunk loop stays in i32 (no i64↔i32 truncation per offset)
    num_chunks = Int32((n_rows + TILE_SIZE - Int32(1)) ÷ Int32(TILE_SIZE))
    row_offsets_base = ct.arange(TILE_SIZE)
    col_tile = ct.Tile(col_idx)

    col_max = fill(-Inf32, (1,))
    denominator = zeros(Float32, TILE_SIZE)

    # Pass 1: Find maximum across all chunks
    for chunk_idx in Int32(0):num_chunks - Int32(1)
        row_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ row_offsets_base
        chunk = ct.gather(input, (row_indices, col_tile);
                         check_bounds=true, padding_value=T(-Inf))
        chunk = convert(ct.Tile{Float32}, chunk)
        chunk_max = maximum(chunk)
        col_max = max.(col_max, ct.Tile(chunk_max))
    end

    # Pass 2: Compute denominator (sum of all exp values)
    for chunk_idx in Int32(0):num_chunks - Int32(1)
        row_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ row_offsets_base
        chunk = ct.gather(input, (row_indices, col_tile);
                         check_bounds=true, padding_value=T(-Inf))
        chunk = convert(ct.Tile{Float32}, chunk)
        numerator = exp.(chunk .- col_max)
        denominator = denominator .+ numerator
    end
    denom_sum = ct.Tile(sum(denominator))

    # Pass 3: Compute final softmax and scatter
    for chunk_idx in Int32(0):num_chunks - Int32(1)
        row_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ row_offsets_base
        chunk = ct.gather(input, (row_indices, col_tile);
                         check_bounds=true, padding_value=T(-Inf))
        chunk = convert(ct.Tile{Float32}, chunk)
        softmax_output = exp.(chunk .- col_max) ./ denom_sum
        ct.scatter(output, (row_indices, col_tile), convert(ct.Tile{T}, softmax_output);
                  check_bounds=true)
    end
    return
end

#=============================================================================
 Host Functions — take (N, M) matrices, softmax over dim 1.
 Launches are asynchronous; Array(output) waits for completion.
=============================================================================#

"""
    softmax_tma!(output, input; tile_size)

TMA single-tile strategy. tile_size must be >= size(input, 1).
"""
function softmax_tma!(output::CuMatrix{T}, input::CuMatrix{T};
                      tile_size::Int=nextpow(2, size(input, 1))) where {T}
    M = size(input, 2)
    # Tiles up to 2048 run best at occupancy 8; larger tiles need the registers.
    occupancy = tile_size <= 2048 ? 8 : 4
    @cuda backend=cuTile blocks=M occupancy=occupancy softmax_kernel_tma(output, input, ct.Constant(tile_size))
    return
end

"""
    softmax_online!(output, input; tile_size)

Online softmax strategy. Processes each column in tile_size chunks.
"""
function softmax_online!(output::CuMatrix{T}, input::CuMatrix{T};
                         tile_size::Int=min(2048, nextpow(2, size(input, 1)))) where {T}
    M = size(input, 2)
    @cuda backend=cuTile blocks=M softmax_kernel_online(output, input, ct.Constant(tile_size))
    return
end

"""
    softmax_chunked!(output, input; tile_size)

Chunked softmax strategy (3-pass, gather/scatter).
"""
function softmax_chunked!(output::CuMatrix{T}, input::CuMatrix{T};
                          tile_size::Int=min(1024, nextpow(2, size(input, 1)))) where {T}
    N, M = size(input)
    @cuda backend=cuTile blocks=M softmax_kernel_chunked(output, input, ct.Constant(N), ct.Constant(tile_size))
    return
end

#=============================================================================
 Verification
=============================================================================#

# Reference over dim 1 of an (N, M) matrix
function ref_softmax(inp::Matrix{Float32})
    col_max = maximum(inp; dims=1)
    exp_vals = exp.(inp .- col_max)
    return exp_vals ./ sum(exp_vals; dims=1)
end

function test_tma(M, N, TILE_SIZE)
    println("  Strategy 1: TMA single-tile ($M vectors × $N, tile=$TILE_SIZE)")
    inp = CUDA.randn(Float32, N, M)
    out = CUDA.zeros(Float32, N, M)

    softmax_tma!(out, inp; tile_size=TILE_SIZE)

    expected = ref_softmax(Array(inp))
    @assert isapprox(Array(out), expected; rtol=1e-3, atol=1e-3) (
        "TMA mismatch! max diff: $(maximum(abs.(Array(out) .- expected)))"
    )
    println("    PASSED")
end

function test_online(M, N, TILE_SIZE)
    println("  Strategy 2: Online 2-pass ($M vectors × $N, tile=$TILE_SIZE)")
    inp = CUDA.randn(Float32, N, M)
    out = CUDA.zeros(Float32, N, M)

    softmax_online!(out, inp; tile_size=TILE_SIZE)

    expected = ref_softmax(Array(inp))
    @assert isapprox(Array(out), expected; rtol=1e-3, atol=1e-3) (
        "Online mismatch! max diff: $(maximum(abs.(Array(out) .- expected)))"
    )
    println("    PASSED")
end

function test_chunked(M, N, TILE_SIZE)
    println("  Strategy 3: Chunked 3-pass ($M vectors × $N, tile=$TILE_SIZE)")
    inp = CUDA.randn(Float32, N, M)
    out = CUDA.zeros(Float32, N, M)

    softmax_chunked!(out, inp; tile_size=TILE_SIZE)

    expected = ref_softmax(Array(inp))
    @assert isapprox(Array(out), expected; rtol=1e-3, atol=1e-3) (
        "Chunked mismatch! max diff: $(maximum(abs.(Array(out) .- expected)))"
    )
    println("    PASSED")
end

function main()
    println("=== cuTile.jl Softmax Examples (3 strategies) ===\n")

    test_tma(256, 512, 512)
    test_online(256, 4096, 1024)
    test_chunked(256, 4096, 256)

    println("\n=== All softmax examples completed ===")
end

isinteractive() || main()
