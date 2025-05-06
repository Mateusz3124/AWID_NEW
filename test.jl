using Pkg
Pkg.develop(path="./differential/")
using differential: forward!, backward!, Variable, topological_sort, Constant

function autograd(f, args...)
    vars = map(x -> Variable(Float32(x)), args)

    result = f(vars...)

    graph = topological_sort(result)

    forward!(graph)
    backward!(graph)

    grads = map(v -> v.gradient, vars)

    return result.output, grads
end

macro test(expr)
    transformed = _replace_literal(expr)
    return esc(transformed)
end

function _replace_literal(expr)
    if expr isa Expr
        return Expr(expr.head, map(_replace_literal, expr.args)...)
    elseif expr isa Number
        return :(Constant(Float32($expr)))
    else
        return expr
    end
end

const1 = Constant(2)
const2 = Constant(5)

func = @test (x, y) -> const1 * x .+ const2 * y .+ 2

println(autograd(func, 2, 4))
