cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using RxInfer
using Random
using StableRNGs

using LinearAlgebra     # only used for some matrix specifics
using Plots             # only used for visualisation
using Distributions     # only used for sampling from multivariate distributions
using Optim             # only used for parameter optimisation 
using SequentialSamplingModels

model = FlowModel(2,
    (
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
);

compiled_model = compile(model, randn(StableRNG(321), nr_params(model)))

function generate_data(nr_samples::Int64, model::CompiledFlowModel)

    
    # specify latent sampling distribution
    dist = LCA(; α = 1.5, β=0.20, λ=0.10, ν=[2.5,2.0], Δt=.001, τ=.30, σ=1.0)


    # sample from the distribution
    choice,rt = rand(dist, nr_samples)

    # transform data
    y = zeros(Float64, 2, nr_samples)
    for k = 1:nr_samples
        y[:,k] .= ReactiveMP.forward(model, [choice[k], rt[k]])
    end

    # return data
    return y, [choice rt]'

end;
# generate data
y, x = generate_data(1000, compiled_model)

# plot generated data
p1 = scatter(x[1,:], x[2,:], alpha=0.3, title="Original data", size=(800,400))
p2 = scatter(y[1,:], y[2,:], alpha=0.3, title="Transformed data", size=(800,400))
plot(p1, p2, legend = false)


@model function invertible_neural_network(nr_samples::Int64)
    
    # initialize variables
    z_μ   = randomvar()
    z_Λ   = randomvar()
    x     = randomvar(nr_samples)
    y_lat = randomvar(nr_samples)
    y     = datavar(Vector{Float64}, nr_samples)

    # specify prior
    z_μ ~ MvNormalMeanCovariance(zeros(2), huge*diagm(ones(2)))
    z_Λ ~ Wishart(2.0, tiny*diagm(ones(2)))

    # specify observations
    for k = 1:nr_samples

        # specify latent state
        x[k] ~ MvNormalMeanPrecision(z_μ, z_Λ)

        # specify transformed latent value
        y_lat[k] ~ Flow(x[k])

        # specify observations
        y[k] ~ MvNormalMeanCovariance(y_lat[k], tiny*diagm(ones(2)))

    end

    # return variables
    return z_μ, z_Λ, x, y_lat, y

end;


observations = [y[:,k] for k=1:size(y,2)]

fmodel         = invertible_neural_network(length(observations))
data          = (y = observations, )
initmarginals = (z_μ = MvNormalMeanCovariance(zeros(2), huge*diagm(ones(2))), z_Λ = Wishart(2.0, tiny*diagm(ones(2))))
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