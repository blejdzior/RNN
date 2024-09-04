module NeuralNetwork

export RNNLayer, Dense, RNN, reset_net!, Descent, loader, loss_and_accuracy, apply!, update!, loss, forward!, backward!, prediction, topological_sort, train!
using ProgressMeter, InteractiveUtils, BenchmarkTools
using Distributions, Flux
using Statistics: mean
include("autodiff.jl")
using .AutoDiff

import Base.show
show(x::AbstractVecOrMat{T}) where {T} = show(IOContext(stdout, :limit => true), "text/plain", x), println(stdout)

# Xavier initialization
function xav_init(dims::Integer...)
    xav = sqrt(6/(sum(dims)))
    return rand(Uniform(-xav, xav), dims...)
end


mutable struct RNNLayer{F, G, S, B}
    σ::F
    Wi::G
    Wh::S
    b::B
    state0::Matrix{Float64} # initial state of RNN Cell
    
    function RNNLayer((in, out)::Pair{<:Integer, <:Integer}, σ = tanh)
        Wi = Variable(xav_init(out, in), name="Wi")
        Wh = Variable(xav_init(out, out), name="Wh")
        b = Variable(zeros(out), name="b")
        state0 = zeros(out, 1)
        new{typeof(σ), typeof(Wi), typeof(Wh), typeof(b)}(σ, Wi, Wh, b, state0)
    end
end



# RNNLayer forward pass
function (i::RNNLayer)(x::Variable, state::Variable)
    Wi, Wh, b = i.Wi, i.Wh, i.b
    return i.σ.(Wi * x .+ Wh * state .+ b)
end

function (i::RNNLayer)(x::AbstractVecOrMat, state::Variable)
    x = Variable(x, name="x")
    return i(x, state)
end

function(i::RNNLayer)(x::Variable, y::GraphNode)
    return i.σ.(i.Wi * x .+ i.Wh * y .+ i.b)
end

function (i::RNNLayer)(x::AbstractVecOrMat, y::GraphNode)
    x = Variable(x, name="x")
    return i(x, y)
end

mutable struct Dense{F, G, S}
    σ::F
    Why::G
    b::S
   

    function Dense((in, out)::Pair{<:Integer, <:Integer}, σ = identity)
        # Xavier initialization 
        Why = Variable(xav_init(out, in), name="Why")
        # bias initialization
        b = Variable(fill!(similar(Why.output, size(Why.output, 1)), zero(eltype(Why.output))), name="b")
        # new object
        new{typeof(σ), typeof(Why), typeof(b)}(σ, Why, b)
    end
end

# forward pass for Dense layer
function(i::Dense)(x::Variable)
    return i.σ.(i.Why * x .+ i.b)
end

function(i::Dense)(x::BroadcastedOperator)
    return i.σ.(i.Why * x .+ i.b)
end

mutable struct RNN{R, D}
    RNNCell::R
    dense::D
    state::Variable
    grads::Vector{VecOrMat{Float64}}
    function RNN(cell::R, dense::D) where {R, D}
        state = Variable(cell.state0, name="state")
        new{R, D}(cell, dense, state, [similar(dense.Why.output), similar(dense.b.output), similar(cell.Wi.output), similar(cell.Wh.output), similar(cell.b.output)])
    end
end

# Calling forward passess for both layers
function (i::RNN)(x::AbstractVecOrMat)
    x = Variable(x, name="x")
    order_state = i.RNNCell(x, i.state)

    i.state.output = forward!(topological_sort(order_state))
    # show(i.state.output)
    # sleep(100)
    order = i.dense(order_state)
    return  order
end

function reset_net!(i::RNN)
    i.state.output = i.RNNCell.state0
end


# struct defining gradient descent optimiser
struct Descent
    eta::Float64
end

function apply(i::Descent)
    return  i.eta
end

function loader(data; batchsize::Int=1)
    x1dim = reshape(data.features, 28 * 28, :) # reshape 28×28 pixels into a vector of pixels
    yhot  = Flux.onehotbatch(data.targets, 0:9) # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x1dim, yhot); batchsize, shuffle=true)
end

function loss_and_accuracy(model, data)
    (x,y) = only(loader(data; batchsize=length(data)))
    reset_net!(model)
    x_1 = @views x[1:196, :]
    ŷ_order = model.RNNCell(x_1, model.state)

    x_1 = @view x[197:392, :]
    ŷ_order = model.RNNCell(x_1, ŷ_order)


    x_1 = @view x[393:588, :]
    ŷ_order = model.RNNCell(x_1, ŷ_order)


    x_1 = @view x[589:end, :]
    ŷ_order = model.RNNCell(x_1, ŷ_order)

    ŷ_order = model.dense(ŷ_order)

    y_1 = @view y[:, :]
    loss_order = topological_sort(NeuralNetwork.loss(ŷ_order, y_1))
    loss = forward!(loss_order)
    ŷ = forward!(topological_sort(ŷ_order))
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    (; loss, acc, split=data.split)  # return a NamedTuple
end


function update!(net, optimiser::Descent)
    η = apply(optimiser)

    net.dense.Why.output .=  net.dense.Why.output .-  η  .* net.grads[1]
    net.dense.b.output .= net.dense.b.output .- η  .* net.grads[2]
    net.RNNCell.Wi.output .= net.RNNCell.Wi.output .-  η  .* net.grads[3]
    net.RNNCell.Wh.output .= net.RNNCell.Wh.output  .-  η  .* net.grads[4]
    net.RNNCell.b.output .=  net.RNNCell.b.output .- η  .* net.grads[5]
    return nothing
end



function loss(model, x::AbstractVecOrMat, y::AbstractVecOrMat)
    y = Variable(y, name="y")
    return AutoDiff.crossentropy(AutoDiff.softmax(model(x)), y)
end

function loss(x::GraphNode, y::AbstractVecOrMat)
    y = Variable(y, name="y")
    return AutoDiff.crossentropy(AutoDiff.softmax(x), y)
end

function loss(x::GraphNode, y::GraphNode)
    return AutoDiff.crossentropy(AutoDiff.softmax(x), y)
end

function prediction(model, x)
    return model(x)
end


function forward!(Order::Vector)
    return AutoDiff.forward!(Order)
end

function backward!(Order::Vector)
    AutoDiff.backward!(Order)
    return nothing
end

function update_grads!(model)

    model.grads[1] .+= model.dense.Why.gradient
    model.grads[2] .+= model.dense.b.gradient
    model.grads[3] .+= model.RNNCell.Wi.gradient
    model.grads[4] .+= model.RNNCell.Wh.gradient
    model.grads[5] .+= model.RNNCell.b.gradient
    return nothing
end

function reset_grads!(model)
    @inbounds for i in eachindex(model.grads)
        fill!(model.grads[i], zero(eltype(model.grads[i])))
    end
    return nothing
end

function mean_grads!(model, batchsize)
    for grads in model.grads
        grads ./= batchsize
    end
end


function train!(model, optimiser, x::Matrix{Float32}, y::AbstractArray; batchsize::Int64=1)
    reset_grads!(model)
   @simd for i = 1:batchsize
        y_1 = view(y, :, i)
        
        x_1 = @view x[1:196, i]
        ŷ_order = model.RNNCell(x_1, model.state)
        x_1 = @view x[197:392, i]
        ŷ_order = model.RNNCell(x_1, ŷ_order)
   
        x_1 = @view x[393:588, i]
        ŷ_order = model.RNNCell(x_1, ŷ_order)

        x_1 = @view x[589:end, i]
        ŷ_order = model.RNNCell(x_1, ŷ_order)

        ŷ_order = model.dense(ŷ_order)
        
        loss_order = topological_sort(NeuralNetwork.loss(ŷ_order, y_1))
        loss = forward!(loss_order)
        backward!(loss_order)
        update_grads!(model)
    end
    update!(model, optimiser)

    return nothing
end

end