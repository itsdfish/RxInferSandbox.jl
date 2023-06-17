cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using RxInfer
using ReactiveMP        # using branch transfominator
using Random
using StableRNGs

using LinearAlgebra     # only used for some matrix specifics
using Plots             # only used for visualisation
using Distributions     # only used for sampling from multivariate distributions
using Optim             # only used for parameter optimisation 
using SequentialSamplingModels

model = FlowModel(
    (
        InputLayer(2),
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
);

compiled_model = compile(model, randn(StableRNG(321), nr_params(model)))

function generate_data(nr_samples::Int64, prior_samples, model::CompiledFlowModel)
    # training data. rows choices, reaction times
    y = zeros(Float64, 2, nr_samples * prior_samples)
    col = 0
    for p ∈ 1:prior_samples
        α = rand(Uniform(0, 3))
        τ = rand(Uniform(0, 0.50))
        # specify latent sampling distribution
        dist = LCA(; α, β=0.20, λ=0.10, ν=[2.5,2.0], Δt=.001, τ, σ=1.0)
        for k ∈ 1:nr_samples
            col += 1
            # sample from the distribution
            choice,rt = rand(dist)
            y[:,col] = [choice, rt] 
        end
    end
    return y
end

# generate data
y = generate_data(100, 100, compiled_model)

# plot generated data
p1 = scatter(y[1,:], y[2,:], alpha=0.3, title="Original data", size=(800,400))

ατconcat(α, τ) = vcat(α, τ) # we need this function to combine α and τ parameters and push it through the Flow

@model function invertible_neural_network(nr_samples::Int64, model)
    
    # initialize variables
    x     = randomvar(nr_samples)
    y_lat = randomvar(nr_samples)
    y     = datavar(Vector{Float64}, nr_samples)

    # specify prior
    α  ~ Normal(μ=3, σ²=10.0)
    τ  ~ Normal(μ=10, σ²=10.0)

    # α   ~ Beta(1, 1)
    # τ   ~ Beta(1, 1)

    # use the following for Beta priors 
    #z_μ ~ ατconcat(α, τ) where {meta=CVI()}
    z_μ ~ ατconcat(α, τ) where {meta=Linearization()}
    z_Λ ~ Wishart(1e2, 1e4 * diageye(2))

    # specify observations
    for i in 1:nr_samples

        # specify latent state
        x[i] ~ MvNormal(μ=z_μ, Λ=z_Λ)

        # specify transformed latent value
        y_lat[i] ~ Flow(x[i]) where {meta=FlowMeta(model)}

        # specify observations
        y[i] ~ MvNormal(μ=y_lat[i], Σ=tiny * diageye(2))

    end
end


fmodel         = invertible_neural_network(length(y), compiled_model)
data          = (y = y, )
initmarginals = (z_μ = MvNormalMeanCovariance(zeros(2), huge*diageye(2)), z_Λ = Wishart(2.0, diageye(2)))
returnvars    = (z_μ = KeepLast(), z_Λ = KeepLast(), x = KeepLast(), y_lat = KeepLast())

constraints = @constraints begin
    q(z_μ, x, z_Λ) = q(z_μ)q(z_Λ)q(x)
end

@meta function fmeta(model)
    compiled_model = compile(model, randn(StableRNG(321), nr_params(model)))
    Flow(y_lat, x) -> FlowMeta(compiled_model) # defaults to FlowMeta(compiled_model; approximation=Linearization()). 
                                               # other approximation methods can be e.g. FlowMeta(compiled_model; approximation=Unscented(input_dim))
end

# First execution is slow due to Julia's initial compilation 
result = inference(
    model = fmodel, 
    data  = data,
    constraints   = constraints,
    meta          = fmeta(model),
    initmarginals = initmarginals,
    returnvars    = returnvars,
    free_energy   = true,
    iterations    = 10, 
    showprogress  = false
)

fe_flow = result.free_energy
zμ_flow = result.posteriors[:z_μ]
zΛ_flow = result.posteriors[:z_Λ]
x_flow  = result.posteriors[:x]
y_flow  = result.posteriors[:y_lat];


plot(1:10, fe_flow/size(y,2), xlabel="iteration", ylabel="normalized variational free energy [nats/sample]", legend=false)
