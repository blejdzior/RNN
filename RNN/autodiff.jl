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
   return nothing
end

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

crossentropy(x::GraphNode, y::GraphNode) = BroadcastedOperator{Float64}(crossentropy, x, y)
forward(::BroadcastedOperator{typeof(crossentropy)}, x, y) = return mean(-sum(y .* log.(x .+ eps(eltype(x)))))
backward(node::BroadcastedOperator{typeof(crossentropy)}, x, y, g) = let 
    @. x.gradient +=  g * (x.output - y.output)
    return nothing
end
end
