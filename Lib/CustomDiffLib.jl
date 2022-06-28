# Definicja liczb dualnych oraz niezbędnych przeciążeń
# Do implementacji algorytmu różniczkowania w przód


import Base: convert, promote_rule
# Zdefiniowanie struktury


struct Dual{T<:Number} <: Number
    v::T
    dv::T
end

# Przeciążenie podstawowych operatorów

import Base: +, -, *, /

-(x::Dual) = Dual(-x.v, -x.dv)
+(x::Dual, y::Dual) = Dual(x.v + y.v, x.dv + y.dv)
-(x::Dual, y::Dual) = Dual(x.v - y.v, x.dv - y.dv)
*(x::Dual, y::Dual) = Dual(x.v * y.v, x.dv * y.v + x.v * y.dv)
/(x::Dual, y::Dual) = Dual(x.v / y.v, (x.dv * y.v - x.v * y.dv) / y.v^2)

# Przeciążenie podstawowych funkcji

import Base: abs, sin, cos, tan, exp, sqrt, isless
abs(x::Dual) = Dual(abs(x.v), sign(x.v) * x.dv)
sin(x::Dual) = Dual(sin(x.v), cos(x.v) * x.dv)
cos(x::Dual) = Dual(cos(x.v), -sin(x.v) * x.dv)
tan(x::Dual) = Dual(tan(x.v), one(x.v) * x.dv + tan(x.v)^2 * x.dv)
exp(x::Dual) = Dual(exp(x.v), exp(x.v) * x.dv)
sqrt(x::Dual) = Dual(sqrt(x.v), 0.5 / sqrt(x.v) * x.dv)
isless(x::Dual, y::Dual) = x.v < y.v;

convert(::Type{Dual{T}}, x::Dual) where {T} = Dual(convert(T, x.v), convert(T, x.dv))
convert(::Type{Dual{T}}, x::Number) where {T} = Dual(convert(T, x), zero(T))
promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} = Dual{promote_type(T, R)}


import Base: show
show(io::IO, x::Dual) = print(io, "(", x.v, ") + [", x.dv, "ϵ]");
value(x::Dual) = x.v;
partials(x::Dual) = x.dv;

D = derivative(f, x) = partials(f(Dual(x, one(x))))


J = function jacobian(f, args::Vector{T}) where {T<:Number}
    jacobian_columns = Matrix{T}[]

    for i = 1:length(args)
        x = Dual{T}[]
        for j = 1:length(args)
            seed = (i == j)
            push!(x, seed ?
                     Dual(args[j], one(args[j])) :
                     Dual(args[j], zero(args[j])))
        end
        column = partials.([f(x)...])
        push!(jacobian_columns, column[:, :])
    end
    hcat(jacobian_columns...)
end