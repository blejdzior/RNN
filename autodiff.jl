module AutoDiff
using LinearAlgebra
using Statistics: mean
using InteractiveUtils, BenchmarkTools

export Variable, BroadcastedOperator, backward!, forward!, topological_sort, softmax, crossentropy, GraphNode
# Structures
abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable{T, G} <: GraphNode
    output :: T
    gradient :: G
    name :: String
    Variable(output::T; name="?") where {T} = let
        if size(output, 2) === 1 && typeof(output) != Matrix{Float64}
            var = new{T, Vector{Float64}}()
        else
            var = new{T, Matrix{Float64}}()
        end
        var.output = output
        var.name  = name
        var.gradient = zeros(Float64, size(output)) 
        return var
    end
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F, T} <: Operator
    inputs :: Tuple{Vararg{GraphNode}}
    output :: T
    gradient :: T
    name :: String
    BroadcastedOperator{T}(fun, inputs...; name="?") where{T} = let 
        op = new{typeof(fun), T}()
        op.inputs = inputs
        op.output = forward(op,[input.output for input in op.inputs]... )
        if length(op.output) > 1
            op.gradient = similar(op.output)
        else 
            op.gradient = zero(T)
        end
        op.name = name
        return op
    end
end

# mutable struct BroadcastedOperator{F, T} <: Operator
#     inputs :: Tuple{Vararg{GraphNode}}
#     output :: T
#     gradient :: T
#     name :: String
#     function BroadcastedOperator{T}(fun::F, inputs...; name="?") where {T, F}
#         op = new{F, T}()
#         op.inputs = inputs
#         input_outputs = [input.output for input in inputs]

#         # Determine the output shape and type based on the function
#         output_shape, output_type = determine_output_shape_type(typeof(fun), input_outputs)

#         # Preallocate output and gradient
#         if isempty(output_shape)
#             op.output = zero(output_type)
#             op.gradient = zero(output_type)
#         else
#             if T == Vector{Float64}
#                 output_shape = output_shape[1]
#             elseif T == Matrix{Float64} && length(output_shape) == 1
#                 output_shape = (output_shape[1], 1)
#             end
#             op.output = zeros(output_type, output_shape)
#             op.gradient = zeros(output_type, output_shape)
#         end
#         show(typeof(fun))
#         println()
#         @time forward(op,[input.output for input in op.inputs]... )
#         op.name = name
#         return op
#     end
# end

# function determine_output_shape_type(type, input_outputs)
#     # Determine the shape and type of the output based on the operation
#     if type == typeof(tanh) || type == typeof(identity)
#         output_type = eltype(input_outputs[1])
#         output_shape = size(input_outputs[1])
#     elseif type == typeof(crossentropy)
#         output_type = Float64
#         output_shape = ()
#     elseif type == typeof(LinearAlgebra.mul!)
#         output_type = eltype(input_outputs[1])
#         output_shape = (size(input_outputs[1], 1), size(input_outputs[2], 2))
#     elseif type == typeof(+) || type == typeof(-)
#         output_type =  eltype(input_outputs[1])
#         output_shape = size(input_outputs[1])
#     elseif type == typeof(softmax)
#         output_type = eltype(input_outputs[1])
#         output_shape = size(input_outputs[1])
#     else
#         throw("Unsupported operation")
#     end
#     return output_shape, output_type
# end

# Pretty Printing 
import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end
show(x::AbstractVecOrMat{T}) where {T} = show(IOContext(stdout, :limit => true), "text/plain", x), println(stdout)


# Graph Building
function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(head, visited, order)
    return order
end

# Forward pass
reset!(node::Constant) = nothing
reset!(node::Variable) = let
    if isnothing(node.gradient)
        if length(node.output) > 1
            node.gradient = zeros(eltype(node.gradient), size(node.output))
        else
            node.gradient = zero(eltype(node.output))
        end
    else
        fill!(node.gradient, zero(eltype(node.gradient)))
    end
end
reset!(node::Operator) = let
    if isnothing(node.gradient) 
        if length(node.output) > 1
            node.gradient = fill!(similar(node.output), zero(eltype(node.output)))
        else
            node.gradient = zero(eltype(node.output))
        end
    else
        if length(node.output) > 1
            fill!(node.gradient, zero(eltype(node.gradient)))
        else
            node.gradient = zero(eltype(node.output))
        end
    end
end

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = if isnothing(node.output)
    node.output = forward(node, [input.output for input in node.inputs]...) else node.output = forward(node, [input.output for input in node.inputs]...) end

function forward!(order::Vector{GraphNode})
    for node in order
        # compute!(node)
        reset!(node)

    end
    return last(order).output
end


function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end


function backward!(node::Operator)
    inputs = node.inputs
    backward(node, inputs..., node.gradient)
    return nothing
end

# Scalar operators
# import Base: ^
# ^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
# forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
# backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

# import Base: sin
# sin(x::GraphNode) = ScalarOperator(sin, x)
# forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
# backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

# Broadcasted operators

import Base: *
import LinearAlgebra: mul!
# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = let 
    if size(x.output, 2) === 1 && typeof(x.output) != Matrix{Float64}
        BroadcastedOperator{Vector{Float64}}(mul!, A, x)
    else 
        BroadcastedOperator{Matrix{Float64}}(mul!, A, x)
    end
end
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = let 
    # A.gradient .+= g * x.output'
    @inbounds for j in axes(A.gradient, 2)  # Loop over columns of A.gradient
        for i in axes(A.gradient, 1) # Loop over rows of A.gradient
            A.gradient[i, j] += g[i] * x.output[j]
        end
    end
    mul!(x.gradient, A.output', g, true, true) 
    # x.gradient .+= A.output' * g
    # @inbounds for j in eachindex(x.gradient)  # Loop over elements of x.gradient
    #     for i in eachindex(g)  # Loop over elements of g
    #         x.gradient[j] += A.output[i, j] * g[i]
    #     end
    # end
   return nothing
end
# x .* y (element-wise multiplication)
# Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator{Matrix{Float64}}(*, x, y)
# forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
# backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
#     # x.gradient .+= g * y.output
#     # y.gradient .+= g * x.output
#     mul!(x.gradient, g, y.output, true, true)
#     mul!(y.gradient, g, x.output, true, true)

#     return nothing
# end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator{Matrix{Float64}}(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = let
    @. x.gradient += g
    @. y.gradient += -g
    return nothing
end

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator{Matrix{Float64}}(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = let 
    @. x.gradient += g
    @. y.gradient += g
    return nothing
end

# import Base: sum
# sum(x::GraphNode) = BroadcastedOperator(sum, x)
# forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
# backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
#     𝟏 = ones(length(x))
#     tuple(𝟏 .* g)
# end

# Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
# forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
# backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = let
#     Jx = 1.0 ./ y
#     Jy = -x ./ (y .^ 2)
#     # Element-wise multiplication with the gradient
#     tuple(g .* Jx, g .* Jy)
# end

# import Base: max
# Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
# forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
# backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
#     Jx = (x .>= y) .* 1.0  # Gradient flows where x is greater than or equal to y
#     Jy = (y .> x) .* 1.0   # Gradient flows where y is greater than x
#     tuple(g .* Jx, g .* Jy)
# end


Base.Broadcast.broadcasted(tanh, x::GraphNode) = BroadcastedOperator{Matrix{Float64}}(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) = return tanh.(x)
backward(node::BroadcastedOperator{typeof(tanh)}, x, g) = let 
    y = node.output
    @. x.gradient += g * (1 - y ^ 2)
    return nothing
end


Base.Broadcast.broadcasted(identity, x::GraphNode) = BroadcastedOperator{Matrix{Float64}}(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = return x
backward(node::BroadcastedOperator{typeof(identity)}, x, g) = let 
    @. x.gradient += g
    return nothing
end

softmax(x::GraphNode) = BroadcastedOperator{Matrix{Float64}}(softmax, x)
Base.Broadcast.broadcasted(softmax, x::GraphNode) = BroadcastedOperator{Matrix{Float64}}(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = let
    exp_x = exp.(x .- maximum(x)) 
    return exp_x ./ sum(exp_x)
end 
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = let
    y = node.output
    @. x.gradient = y * (g - (sum(g * y)))
    return nothing    
end

# mean(-sum(y .* log.(x .+ eps(eltype(x)))))
crossentropy(x::GraphNode, y::GraphNode) = BroadcastedOperator{Float64}(crossentropy, x, y)
forward(::BroadcastedOperator{typeof(crossentropy)}, x, y) = return mean(-sum(y .* log.(x .+ eps(eltype(x)))))
backward(node::BroadcastedOperator{typeof(crossentropy)}, x, y, g) = let 
    @. x.gradient +=  g * (x.output - y.output)
    return nothing
end
end
