# Machine learning
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
# Statistic
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
# Datasets and model storage
import MLDatasets
import BSON
# Image processing
using Images, Colors
# Use python for gui
using PyCall

## GUI

@pyimport tkinter as tk
pil = pyimport_conda("PIL", "pillow")
imggrab = pyimport_conda("PIL.ImageGrab", "pillow")
win32gui = pyimport_conda("win32gui", "pywin32")

@pydef mutable struct Application <: tk.Tk


    function predict_image(self)
        canvas_id = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(canvas_id)
        image = imggrab.grab(rect)
        image.save("temp.jpg")
        self.text.set(predict_number("./temp.jpg"))
    end

    function clear_canvas(self)
        self.canvas.delete("all")
    end

    function draw_oval(self, event)
        self.x = event.x
        self.y = event.y
        radius = 8
        self.canvas.create_oval(self.x - radius, self.y - radius, self.x + radius, self.y + radius, fill="black")
    end

    __init__(self, args...; kwargs...) = begin
        tk.Tk.__init__(self, args...; kwargs...)

        self.text = tk.StringVar()
        self.text_readme = tk.StringVar()
        self.text.set("Press 'Predict' button")
        self.text_readme.set("First prediction will be long")
        self.x = self.y = 0

        self.canvas = tk.Canvas(width=300, height=300, bg="white")
        self.label = tk.Label(textvariable=self.text, font=("Arial", 25))
        self.label_readme = tk.Label(textvariable=self.text_readme, font=("Arial", 12))
        self.save_btn = tk.Button(text="Predict", command=self.predict_image)
        self.сlear_btn = tk.Button(text="Clear", command=self.clear_canvas)

        self.canvas.bind("<B1-Motion>", self.draw_oval)

        self.label.pack()
        self.canvas.pack()
        self.save_btn.pack()
        self.сlear_btn.pack()
        self.label_readme.pack()
    end
end

## MODEL
# Model(LaNet5)
function LeNet5(; imgsize=(28, 28, 1), nclasses=10)
    out_conv_size = (Int(imgsize[1] / 4) - 3, Int(imgsize[2] / 4) - 3, 16)

    return Chain(
                Conv((5, 5), imgsize[end] => 6, relu),
                MaxPool((2, 2)),
                Conv((5, 5), 6 => 16, relu),
                MaxPool((2, 2)),
                flatten,
                Dense(prod(out_conv_size), 120, relu),
                Dense(120, 84, relu),
                Dense(84, nclasses)
          )
end

# Load datasets
function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)

    return train_loader, test_loader
end

loss(circ, y) = logitcrossentropy(circ, y) # Loss calculation

# loss calculation
function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        circ = model(x)
        l += loss(circ, y) * size(x)[end]
        acc += sum(onecold(circ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l / ntot |> round4, acc = acc / ntot * 100 |> round4)
end

# utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits=4)

# arguments
Base.@kwdef mutable struct Args
    lear_eta = 3e-4      # learning rate
    w_decay = 0          # L2 regularizer param, implemented as weight decay
    batchsize = 128      # batch size
    epochs = 200         # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    infotime = 1 	     # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      # log training with tensorboard
    savepath = "model/"  # results path
end

function predict_number(image_path)
    # Load model
    if isfile("./model/model.bson")
        @info "Loading model"
        BSON.@load "./model/model.bson" model                   # After first load it will be loading faster
    else
        @info "Model not found, Error..."
    end

\
    image = load(image_path)                                    # load image
    resize = imresize(image, (28, 28))                          # resize Image
    img_grayscale = Gray.(resize)                               # Conver to grayscale
    mat = convert(Array{Float64}, img_grayscale)                # Convert to array
    reshaped = reshape(mat, (28, 28, 1, :))                     # Reshape to fit input

    result = argmax(model(reshaped))[1] - 1                     # Get prediction
    rm(image_path)                                              # Remove temp file
    return result
end

## TRAINING
function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    device = cpu
    @info "Training on CPU"


    ## DATA
    train_loader, test_loader = get_data(args)
    @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = LeNet5() |> device
    @info "LeNet5 model: $(num_params(model)) trainable params"

    ps = Flux.params(model)

    opt = ADAM(args.lear_eta)
    if args.w_decay > 0 # add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.w_decay), opt)
    end

    ## LOGGING UTILITIES
    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end

    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end

    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                    circ = model(x)
                    loss(circ, y)
                end

            Flux.Optimise.update!(opt, ps, gs)
        end

        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson")
            let model = cpu(model) #return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end

## Starting point
if abspath(PROGRAM_FILE) == @__FILE__

    if !isfile("./model/model.bson")
        @info "Model not found, training..."
        train()
    end

    app = Application()
    app.mainloop()
end
