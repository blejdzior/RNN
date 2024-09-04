module AutoDiff
using LinearAlgebra
using Statistics: mean
using InteractiveUtils

export Variable, BroadcastedOperator, backward!, forward!, topological_sort, softmax, crossentropy, GraphNode
# Structures
abstract type GraphNode end
abstract type Operator <: GraphNode end

# struct Constant{T} <: GraphNode
#     output :: T
# end

# mutable struct Variable{T} <: GraphNode
#     output :: T
#     gradient :: T   
#     name :: String
#     Variable(output::T; name="?") where {T} = new{T}(output, similar(output), name)
# end

# mutable struct ScalarOperator{F, T} <: Operator
#     inputs :: Tuple{Vararg{GraphNode}}
#     output :: T
#     gradient :: T
#     name :: String
#     function ScalarOperator(fun, inputs...; name="?") where {T} 
#         op = new{typeof(fun), T}()
#         # op.inputs = inputs
#         op.name = name
#         return op
#     end
#     # ScalarOperator(fun, inputs...; name="?") where {T} = new{typeof(fun), T}(inputs, nothing, nothing, name)
# end

# mutable struct BroadcastedOperator{F, T} <: Operator
#     inputs :: Tuple{Vararg{GraphNode}}
#     output :: T
#     gradient :: T
#     name :: String
#     BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun), Any}(inputs, nothing, nothing, name)
# end
struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
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
            node.gradient = fill!(similar(node.output), zero(eltype(node.output)))
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

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

# Backward pass
# update!(node::Constant, gradient) = nothing
# update!(node::GraphNode, gradient) = if isnothing(node.gradient)
#     node.gradient = gradient else node.gradient .+= gradient
# end
# function update!(node::GraphNode, gradient)
#     if isnothing(node.gradient)
#         node.gradient = gradient
#     else
#         node.gradient .+= gradient
#     end
# end

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
# function backward!(node::Operator)
#     inputs = node.inputs
#     gradients = backward(node, [input.output for input in inputs]..., node.gradient)
#      for (input, gradient) in zip(inputs, gradients)
#         update!(input, gradient)
#     end
#     return nothing
# end

function backward!(node::Operator)
    inputs = node.inputs
    # show(node)
    # println()
    # show(inputs[1].gradient)
    # println()
   backward(node, inputs..., node.gradient)
#    for input in inputs
#     show(input.output)
#     println() 
#    end
#    show(node.output)
#    println()
#    show(node.gradient)
#    println()
    # println()
    # sleep(10)
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
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = let 
        mul!(A.gradient, g, x.output', true, true) # true indicates to accumulate
        mul!(x.gradient, A.output', g, true, true) 
   return nothing
end
# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    # x.gradient .+= g * y.output
    # y.gradient .+= g * x.output
    mul!(x.gradient, g, y.output, true, true)
    mul!(y.gradient, g, x.output, true, true)

    return nothing
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = let
    @. x.gradient += g
    @. y.gradient += -g
    return nothing
end

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
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


Base.Broadcast.broadcasted(tanh, x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) = return tanh.(x)
backward(node::BroadcastedOperator{typeof(tanh)}, x, g) = let 
    y = node.output
    @. x.gradient += g * (1 - y ^ 2)
    return nothing
end


Base.Broadcast.broadcasted(identity, x::GraphNode) = BroadcastedOperator(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = return x
backward(node::BroadcastedOperator{typeof(identity)}, x, g) = let 
    @. x.gradient += g
    return nothing
end

softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
Base.Broadcast.broadcasted(softmax, x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = let
    exp_x = exp.(x .- maximum(x)) 
    return exp_x ./ sum(exp_x)
   
end 
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = let
    y = node.output
    @. x.gradient = y * (g - (sum(g * y)))
    return nothing    
end

crossentropy(x::GraphNode, y::GraphNode) = BroadcastedOperator(crossentropy, x, y)
forward(::BroadcastedOperator{typeof(crossentropy)}, x, y) = return mean(-sum(y .* log.(x .+ eps(eltype(x)))))
backward(node::BroadcastedOperator{typeof(crossentropy)}, x, y, g) = let 
    @. x.gradient +=  g * (x.output - y.output)
    return nothing
end
end
