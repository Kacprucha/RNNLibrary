abstract type Optimiser end

struct SGD <: Optimiser
    lr::Float64
end

function update!(opt::SGD, ps, grads)
    for i in eachindex(ps)
        ps[i] .-= opt.lr .* grads[i]
    end
end

struct Adam <: Optimiser
    lr::Vector{Float32}
    beta1::Float32
    beta2::Float32
    eps::Float32
    m::Vector{Array{Float32}}
    v::Vector{Array{Float32}}
    t::Vector{Int}
end

function Adam(lr::Float32, ps; beta1=0.9f0, beta2=0.999f0, eps=1e-8)
    m = [zeros(Float32, size(p)) for p in ps]
    v = [zeros(Float32, size(p)) for p in ps] 
    Adam([lr], beta1, beta2, convert(Float32, eps), m, v, [0])
end

function update!(opt::Adam, ps, grads)
    opt.t[1] += 1
    for i in eachindex(ps)
        opt.m[i] .= opt.beta1 .* opt.m[i] .+ (1 - opt.beta1) .* grads[i]
        opt.v[i] .= opt.beta2 .* opt.v[i] .+ (1 - opt.beta2) .* (grads[i].^2)
        m_hat = opt.m[i] ./ (1 - opt.beta1^opt.t[1])
        v_hat = opt.v[i] ./ (1 - opt.beta2^opt.t[1])
        ps[i] .-= opt.lr[1] .* m_hat ./ (sqrt.(v_hat) .+ opt.eps)
    end
end