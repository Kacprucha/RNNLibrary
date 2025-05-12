import Base: +, -, *, /, sin, cos, tan, cot, sec, csc, exp, log, ^, max
import Statistics: mean
import Base.:-
import Base.:*

# Promote a scalar ReverseNode to an array ReverseNode with the given shape.
function promote_to_array(a::ReverseNode{Float64}, shp::Tuple)
    return ReverseNode(fill(a.value, shp))
end

-(a::Number, b::Array) = a .- b
-(a::Array, b::Number) = a .- b
+(a::Number, b::Array) = a .+ b
+(a::Array, b::Number) = a .+ b

Base.Broadcast.broadcastable(x::ReverseNode) = Ref(x)

# Addition: z = a + b  =>  dz/da = 1, dz/db = 1
+(a::ReverseNode, b::ReverseNode) = begin
    out = ReverseNode(a.value + b.value)
    push!(out.children, (a, δ -> δ))
    push!(out.children, (b, δ -> δ))
    out
end

+(a::ReverseNode{Float64}, b::ReverseNode{T}) where {T<:Array} = begin
    promoted_a = promote_to_array(a, size(b.value))
    return promoted_a + b
end

+(a::ReverseNode{T}, b::ReverseNode{Float64}) where {T<:Array} = begin
    promoted_b = promote_to_array(b, size(a.value))
    return a + promoted_b
end

+(a::ReverseNode{T1}, b::ReverseNode{T2}) where {T1<:Array, T2<:Array} = begin
    out = ReverseNode(a.value .+ b.value)
    push!(out.children, (a, δ -> δ))
    push!(out.children, (b, δ -> δ))
    return out
end

+(a::ReverseNode{T}, b::Number) where {T<:Array} = begin
    out = ReverseNode(a.value .+ b)
    push!(out.children, (a, δ -> δ))
    return out
end

+(a::Number, b::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(a .+ b.value)
    push!(out.children, (b, δ -> δ))
    return out
end

+(a::Array, b::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(a .+ b.value)
    push!(out.children, (b, δ -> δ))
    return out
end

+(a::ReverseNode{T}, b::Array) where {T<:Array} = begin
    out = ReverseNode(a.value .+ b)
    push!(out.children, (a, δ -> δ))
    return out
end

+(a::ReverseNode{T}, b::Number) where {T<:Number} = begin
    out = ReverseNode(a.value + b)
    push!(out.children, (a, δ -> δ))
    return out
end

+(a::Number, b::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(a + b.value)
    push!(out.children, (b, δ -> δ))
    return out
end

# Subtraction: z = a - b  =>  dz/da = 1, dz/db = -1
-(a::ReverseNode, b::ReverseNode) = begin
    out = ReverseNode(a.value - b.value)
    push!(out.children, (a, δ -> δ))
    push!(out.children, (b, δ -> δ * -1.0))
    out
end

# When left is scalar and right is array, promote the scalar.
-(a::ReverseNode{Float64}, b::ReverseNode{T}) where {T<:Array} = begin
    promoted_a = promote_to_array(a, size(b.value))
    return promoted_a - b
end

# When left is array and right is scalar, promote the scalar.
-(a::ReverseNode{T}, b::ReverseNode{Float64}) where {T<:Array} = begin
    promoted_b = promote_to_array(b, size(a.value))
    return a - promoted_b
end

-(a::ReverseNode) = begin
    out = ReverseNode(-a.value)
    push!(out.children, (a, δ -> -δ))
    return out
end

-(a::Number, b::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(a .- b.value)
    push!(out.children, (b, δ -> -δ))
    return out
end

-(a::ReverseNode{T}, b::Number) where {T<:Array} = begin
    out = ReverseNode(a.value .- b)
    push!(out.children, (a, δ -> δ))
    return out
end

-(a::Array, b::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(a .- b.value)
    push!(out.children, (b, δ -> -δ))
    return out
end

-(a::ReverseNode{T}, b::Array) where {T<:Array} = begin
    out = ReverseNode(a.value .- b)
    push!(out.children, (a, δ -> δ))
    return out
end

# Multiplication: z = a * b  =>  dz/da = b, dz/db = a
*(a::ReverseNode{T}, b::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(a.value * b.value)
    push!(out.children, (a, δ -> δ * b.value))
    push!(out.children, (b, δ -> δ * a.value))
    out
end

# Matrix multiplication: z = a * b  =>  dz/da = b', dz/db = a'
*(a::ReverseNode{T1}, b::ReverseNode{T2}) where {T1<:Array, T2<:Array} = begin
    if size(a.value) == size(b.value)
        # Both operands have the same shape: perform elementwise multiplication.
        out_val = a.value .* b.value
        out = ReverseNode(out_val)
        push!(out.children, (a, δ -> δ .* b.value))
        push!(out.children, (b, δ -> δ .* a.value))
        return out
    else
        # Otherwise, assume matrix multiplication is desired.
        out_val = a.value * b.value
        out = ReverseNode(out_val)
        if ndims(a.value) == 2 && ndims(b.value) == 1
            # Matrix multiplication: handle vector cases
            push!(out.children, (a, δ -> (ndims(δ)==1 ? reshape(δ, :, 1) : δ) * transpose(b.value)))
            push!(out.children, (b, δ -> vec(transpose(a.value) * (ndims(δ)==1 ? reshape(δ, :, 1) : δ))))
        else
            push!(out.children, (a, δ -> δ * b.value))
            push!(out.children, (b, δ -> δ * a.value))
        end
        return out
    end
end

*(a::ReverseNode{T}, b::Number) where {T<:Array} = begin
    out = ReverseNode(a.value .* b)
    push!(out.children, (a, δ -> δ .* b))
    return out
end

*(a::Number, b::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(a .* b.value)
    push!(out.children, (b, δ -> δ .* a))
    return out
end

*(a::Array, b::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(a * b.value)
    push!(out.children, (b, δ -> δ * a'))
    return out
end

*(a::ReverseNode{T}, b::Array) where {T<:Array} = begin
    out = ReverseNode(a.value * b)  
    push!(out.children, (a, δ -> δ * b'))
    return out
end


# Division: z = a / b  =>  dz/da = 1/b, dz/db = -a/(b^2)
/(a::ReverseNode, b::ReverseNode) = begin
    out = ReverseNode(a.value / b.value)
    push!(out.children, (a, δ -> δ * (1 / b.value)))
    push!(out.children, (b, δ -> δ * (-a.value / (b.value^2))))
    out
end

/(a::ReverseNode{T}, b::Number) where {T<:Array} = begin
    out = ReverseNode(a.value ./ b)
    push!(out.children, (a, δ -> δ ./ b))
    return out
end 

/(a::Number, b::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(a ./ b.value)
    push!(out.children, (b, δ -> δ .* (-a ./ (b.value.^2))))
    return out
end

/(a::Number, b::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(a / b.value)
    push!(out.children, (b, δ -> δ * (-a / (b.value^2))))
    return out
end

/(a::ReverseNode{T}, b::Number) where {T<:Number} = begin
    out = ReverseNode(a.value / b)
    push!(out.children, (a, δ -> δ * (1 / b)))
    return out
end

# sin: z = sin(a)  =>  dz/da = cos(a)
sin(a::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(sin(a.value))
    push!(out.children, (a, δ -> δ * cos(a.value)))
    out
end

sin(a::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(sin.(a.value))
    push!(out.children, (a, δ -> δ .* cos.(a.value)))
    out
end

# cos: z = cos(a)  =>  dz/da = -sin(a)
cos(a::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(cos(a.value))
    push!(out.children, (a, δ -> δ * -sin(a.value)))
    out
end

cos(a::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(cos.(a.value))
    push!(out.children, (a, δ -> δ .* -sin.(a.value)))
    out
end

# tan: z = tan(a)  =>  dz/da = sec(a)^2
tan(a::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(tan(a.value))
    push!(out.children, (a, δ -> δ * sec(a.value)^2))

    out
end

tan(a::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(tan.(a.value))
    push!(out.children, (a, δ -> δ .* sec.(a.value).^2))
    out
end

# cot: z = cot(a)  =>  dz/da = -csc(a)^2
cot(a::ReverseNode{T}) where{T<:Number} = begin
    out = ReverseNode(cot(a.value))
    push!(out.children, (a, δ -> δ * -csc(a.value)^2))
    out
end

cot(a::ReverseNode{T}) where{T<:Array} = begin
    out = ReverseNode(cot.(a.value))
    push!(out.children, (a, δ -> δ .* -csc.(a.value).^2))
    out
end

# sec: z = sec(a)  =>  dz/da = sec(a)tan(a)
sec(a::ReverseNode{T}) where{T<:Number} = begin
    out = ReverseNode(sec(a.value))
    push!(out.children, (a, δ -> δ * sec(a.value)tan(a.value)))
    out
end

sec(a::ReverseNode{T}) where{T<:Array} = begin
    out = ReverseNode(sec.(a.value))
    push!(out.children, (a, δ -> δ .* sec.(a.value)tan.(a.value)))
    out
end

# csc: z = csc(a)  =>  dz/da = -csc(a)cot(a)
csc(a::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(csc(a.value))
    push!(out.children, (a, δ -> δ * -csc(a.value)cot(a.value)))
    out
end

csc(a::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(csc.(a.value))
    push!(out.children, (a, δ -> δ .* -csc.(a.value)cot.(a.value)))
    out
end

# log: z = log(a)  =>  dz/da = 1/a
log(a::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(log(a.value))
    push!(out.children, (a, δ -> δ /a.value))
    out
end

log(a::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(log.(a.value))
    push!(out.children, (a, δ -> δ ./ a.value))
    out
end

# ^: z = a^b  =>  dz/da = b*a^(b-1), dz/db = a^b*log(a)
^(a::ReverseNode{T}, b::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(a.value ^ b.value)
    push!(out.children, (a, δ -> δ * b.value * a.value^(b.value - 1)))
    push!(out.children, (b, δ -> δ * out.value * log(Complex(a.value))))
    out
end

^(a::ReverseNode{T1}, b::ReverseNode{T2}) where {T1<:Array, T2<:Array} = begin
    out = ReverseNode(a.value .^ b.value)
    push!(out.children, (a, δ -> δ .* b.value .* a.value.^(b.value - 1)))
    push!(out.children, (b, δ -> δ .* out.value .* log.(a.value)))
    out
end

# Overload power for a matrix base and a scalar exponent.
^(a::ReverseNode{Matrix{Float64}}, b::ReverseNode{Float64}) = begin
    out_val = a.value .^ b.value
    out = lift(out_val)
    push!(out.children, (a, δ -> δ .* (b.value .* (a.value .^ (b.value - 1)))))
    push!(out.children, (b, δ -> sum(δ .* (out.value .* log.(max.(a.value, eps()))))))
    out
end

# exp: z = exp(a)  =>  dz/da = exp(a)
exp(a::ReverseNode{T}) where{T<:Number} = begin
    out = ReverseNode(exp(a.value))
    push!(out.children, (a, δ -> δ * exp(a.value)))
    out
end

exp(a::ReverseNode{T}) where{T<:Array} = begin
    out = ReverseNode(exp.(a.value))
    push!(out.children, (a, δ -> δ .* exp.(a.value)))
    out
end

mean(a::ReverseNode) = begin
    out_val = mean(a.value)
    out = lift(out_val)
    n = length(a.value)
    push!(out.children, (a, δ -> fill(δ / n, size(a.value))))
    out
end

max(a::Array, b::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(max.(a, b.value))
    push!(out.children, (b, δ -> δ .* (a .> b.value)))
    return out
end

max(a::Number, b::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(max.(a, b.value))
    push!(out.children, (b, δ -> δ .* (a .> b.value)))
    return out
end

max(a::ReverseNode{T}, b::Number) where {T<:Array} = begin
    out = ReverseNode(max.(a.value, b))
    push!(out.children, (a, δ -> δ .* (a.value .> b)))
    return out
end

max(a::ReverseNode{T}, b::Number) where {T<:Number} = begin
    out = ReverseNode(max(a.value, b))
    push!(out.children, (a, δ -> δ .* (a.value .> b)))
    return out
end

max(a::Number, b::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(max(a, b.value))
    push!(out.children, (b, δ -> δ .* (a .> b.value)))
    return out
end

ReLU(x::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(max(0, x.value))
    push!(out.children, (x, δ -> δ * (x.value .> 0)))
    out
end

ReLU(x::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(max.(0, x.value))
    push!(out.children, (x, δ -> δ .* (x.value .> 0)))
    out
end

Sigmoid(x::ReverseNode{T}) where {T<:Number} = begin
    out = ReverseNode(1 / (1 + exp(-x.value)))
    push!(out.children, (x, δ -> δ * out.value * (1 - out.value)))
    out
end

Sigmoid(x::ReverseNode{T}) where {T<:Array} = begin
    out = ReverseNode(1 ./ (1 .+ exp.(-x.value)))
    push!(out.children, (x, δ -> δ .* out.value .* (1 .- out.value)))
    out
end