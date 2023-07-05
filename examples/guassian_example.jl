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
using Distributions
using Optimisers

model = FlowModel(
    (
        InputLayer(2),
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
);

compiled_model = compile(model, randn(StableRNG(321), nr_params(model)))

# n_samples: number of sampled datasets for training 
# n_max: maximum size of a dataset
function generate_data(n_samples, n_max)
    y = Vector{Vector{Float64}}(undef, n_samples)
    parms = zeros(n_samples, 2)
    for i ∈ 1:n_samples
        μ = rand(Normal(0, 1))
        σ = rand(Gamma(2, 1))
        parms[i,:] = [μ σ]
        n = rand(1:n_max)
        y[i] = rand(Normal(μ, σ), n)
    end
    return y,parms
end

# generate data
y,parms = generate_data(1000, 100)

# # plot generated data
# p1 = scatter(y[1,:], y[2,:], alpha=0.3, title="Original data", size=(800,400))

# ατconcat(α, τ) = vcat(α, τ) # we need this function to combine α and τ parameters and push it through the Flow

# @model function invertible_neural_network(nr_samples::Int64, model)
    
#     # initialize variables
#     x     = randomvar(nr_samples)
#     y_lat = randomvar(nr_samples)
#     y     = datavar(Vector{Float64}, nr_samples)

#     # specify prior

#     μ ~ Normal(0.0, 1.0)
#     σ ~ GammaShapeScale(2, 1)


#     z_μ ~ ατconcat(α, τ) where {meta=CVI(StableRNG(42), 100, 200, Optimisers.Descent(0.1), RxInfer.ForwardDiffGrad(), 100, Val(true), true)}
#     z_Λ ~ Wishart(3, diageye(2))

#     # specify observations
#     for i in 1:nr_samples

#         # specify latent state
#         x[i] ~ MvNormal(μ=z_μ, Λ=z_Λ)

#         # specify transformed latent value
#         y_lat[i] ~ Flow(x[i]) where {meta=FlowMeta(model)}

#         # specify observations
#         y[i] ~ MvNormal(μ=y_lat[i], Σ=tiny*diageye(2))

#     end

#     # return variables
#     return z_μ, z_Λ, x, y_lat, y

# end;


# fmodel         = invertible_neural_network(length(y), compiled_model)
# data          = (y = y, )
# initmarginals = (z_μ = MvNormalMeanCovariance(zeros(2), huge*diageye(2)), z_Λ = Wishart(2.0, diageye(2)))
# returnvars    = (z_μ = KeepLast(), z_Λ = KeepLast(), x = KeepLast(), y_lat = KeepLast())

# constraints = @constraints begin
#     q(z_μ, x, z_Λ) = q(z_μ)q(z_Λ)q(x)
# end

# @meta function fmeta(model)
#     compiled_model = compile(model, randn(StableRNG(321), nr_params(model)))
#     Flow(y_lat, x) -> FlowMeta(compiled_model) # defaults to FlowMeta(compiled_model; approximation=Linearization()). 
#                                                # other approximation methods can be e.g. FlowMeta(compiled_model; approximation=Unscented(input_dim))
# end

# # First execution is slow due to Julia's initial compilation 
# result = inference(
#     model = fmodel, 
#     data  = data,
#     constraints   = constraints,
#     meta          = fmeta(model),
#     initmarginals = initmarginals,
#     returnvars    = returnvars,
#     free_energy   = true,
#     iterations    = 10, 
#     showprogress  = false,
# )

# fe_flow = result.free_energy
# zμ_flow = result.posteriors[:z_μ]
# zΛ_flow = result.posteriors[:z_Λ]
# x_flow  = result.posteriors[:x]
# y_flow  = result.posteriors[:y_lat];


# plot(1:10, fe_flow/size(y,2), xlabel="iteration", ylabel="normalized variational free energy [nats/sample]", legend=false)
