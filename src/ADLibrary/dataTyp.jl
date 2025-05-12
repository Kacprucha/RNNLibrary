struct ReverseNode{T}
    value::T
    grad::Vector{T}
    children::Vector{Tuple{ReverseNode, Function}}
end

ReverseNode(x::Number) = ReverseNode(x,
                                     [zero(x)],
                                     Tuple{ReverseNode, Function}[])

# Constructor for array values:
ReverseNode(x::Array) = ReverseNode(x,
                                    [zeros(eltype(x), size(x))],
                                    Tuple{ReverseNode, Function}[])

lift(x) = ReverseNode(x)

macro diffunction(ex)
    # Check that we have a short-form function definition, e.g. f(x,y) = body
    if ex.head == :(=)
        lhs = ex.args[1]
        rhs = ex.args[2]
        f = lhs.args[1]
        args = lhs.args[2:end]
        new_args = [:( $(arg)::ReverseNode ) for arg in args]
        
        num_literals = Dict{Any,Symbol}()
        
        literal_to_sym(num) = Symbol("lift_" * replace(replace(string(num), "-" => "minus_"), "." => "_"))
        
        # Recursively traverse the expression to replace numbers with variables
        function lift_literals(expr)
            if expr isa Number
                if !haskey(num_literals, expr)
                    num_literals[expr] = literal_to_sym(expr)
                end
                return num_literals[expr]
            elseif expr isa Expr
                return Expr(expr.head, map(lift_literals, expr.args)...)
            else
                return expr
            end
        end
        
        new_rhs = lift_literals(rhs)
        bindings = [:( $(sym) = lift($(float(num))) ) for (num, sym) in num_literals]
        body = Expr(:block, bindings..., :(return $(new_rhs)))

        # Create the new function definition
        new_def = Expr(:function, Expr(:call, f, new_args...), body)
        return esc(new_def)
    else
        error("@diffunction macro must be used with a function definition of the form f(args...) = ...")
    end
end