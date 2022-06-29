using Pkg
libs = [:Random, :Statistics, :RDatasets,
    :DataFrames, :Plots, :TimerOutputs, :Flux, :LinearAlgebra,
    :Combinatorics, :PlutoUI
]

tryusing(pkgsym) =
    try
        @eval using $pkgsym
        return true
    catch e
        Pkg.add(pkgsym)
    end

for lib in libs
    tryusing(lib)
end
