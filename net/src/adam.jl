const EPS = 1e-8

struct Adam
    eta::Float32
    beta::Tuple{Float32,Float32}
    epsilon::Float32
    mt::Matrix{Float32}
    vt::Matrix{Float32}

    function Adam(x::Matrix{Float32}, η::Float32=Float32(0.001), β::Tuple{Float32,Float32}=(Float32(0.9), Float32(0.999)), ϵ::Float32=Float32(1e-8))
        mt = zeros(Float32, size(x))
        vt = zeros(Float32, size(x))
        new(Float32(η), (Float32(β[1]), Float32(β[2])), Float32(ϵ), mt, vt)
    end
end

function apply!(o::Adam, x::Matrix{Float32}, Δ::Matrix{Float32}, t::Int64)
    η, β = o.eta, o.beta
    mt, vt = o.mt, o.vt
    one = Float32(1.0)
    @. mt = β[1] * mt + (one - β[1]) * Δ
    @. vt = β[2] * vt + (one - β[2]) * Δ * conj(Δ)

    mt_hat = mt ./ (one - β[1]^t)
    vt_hat = vt ./ (one - β[2]^t)

    @. x -= η * mt_hat / (√vt_hat + o.epsilon)
end

