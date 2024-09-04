include("NeuralNetwork.jl")
using .NeuralNetwork
using MLDatasets, ProgressMeter, InteractiveUtils
import Base.show
import Flux.onecold
show(x::AbstractVecOrMat{T}) where {T} = show(IOContext(stdout, :limit => true), "text/plain", x), println(stdout)

# Data
train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)


# Setting up the RNN
net = RNN(RNNLayer((14*14) => 64), Dense(64 => 10))

# parameters
train_log = []
settings = (;
    eta = 15e-3,
    epochs = 5,
    batchsize = 100,
)

# Descent optimiser
optimiser = Descent(settings.eta)


@show loss_and_accuracy(net, test_data);  # accuracy about 10%, before training


for epoch in 1:settings.epochs
    
    @time for (x,y) in loader(train_data, batchsize=settings.batchsize)
        reset_net!(net)
        train!(net, optimiser, x, y, batchsize=settings.batchsize)
    end
    loss, acc, _ = loss_and_accuracy(net, train_data)
    test_loss, test_acc, _ = loss_and_accuracy(net, test_data)
    @info epoch acc test_acc
    nt = (; epoch, loss, acc, test_loss, test_acc) 
    push!(train_log, nt)
end


reset_net!(net)
@show loss_and_accuracy(net, train_data);