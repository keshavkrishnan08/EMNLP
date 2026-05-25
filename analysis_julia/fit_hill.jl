# Julia mirror of src/drc/analysis/hill.py. Python is the reference
# implementation; this is provided for the PI's preferred LsqFit/Makie workflow
# and intentionally covers only the core Hill fit (no bootstrap CIs).
#
# Usage:  julia --project=analysis_julia analysis_julia/fit_hill.jl
#
# Reads results/eval_results.csv, fits the four-parameter Hill curve per
# (construction, seed) on the self-eval rows, and writes results/hill_fits.csv.

using CSV
using DataFrames
using LsqFit
using Statistics

const CONSTRUCTIONS = ("aann", "comparative_correlative", "tough_movement", "resultative")

# Fallback attested counts for dose="all", mirroring ATTESTED_ALL_COUNTS in the
# Python module. Keep these in sync if the corpus is regenerated.
const ATTESTED_ALL_COUNTS = Dict(
    "aann" => 1200,
    "comparative_correlative" => 340,
    "tough_movement" => 890,
    "resultative" => 2100,
)

# Four-parameter Hill curve. p = [E0, Emax, E50, n].
function hill(D, p)
    E0, Emax, E50, n = p
    Dc = max.(D, 0.0)
    return E0 .+ (Emax - E0) .* (Dc .^ n) ./ (E50^n .+ (Dc .^ n))
end

"Map dose='all' to the construction's attested count; everything else is numeric."
function dose_value(construction, dose)
    if lowercase(strip(string(dose))) == "all"
        return float(ATTESTED_ALL_COUNTS[construction])
    end
    return parse(Float64, string(dose))
end

function r_squared(Y, Yhat)
    ss_res = sum((Y .- Yhat) .^ 2)
    ss_tot = sum((Y .- mean(Y)) .^ 2)
    ss_tot <= 0 && return 0.0
    return 1.0 - ss_res / ss_tot
end

function fit_cell(D, Y)
    # Initial guess and box constraints matching the Python fit.
    p0 = [minimum(Y), maximum(Y), median(D[D .> 0]), 1.0]
    lower = [0.0, 0.0, 1e-6, 1e-3]
    upper = [1.0, 1.0, Inf, 50.0]
    try
        fit = curve_fit(hill, D, Y, p0; lower=lower, upper=upper)
        Yhat = hill(D, fit.param)
        return (fit.param, r_squared(Y, Yhat), true)
    catch err
        @warn "Hill fit failed" error=err
        return ([NaN, NaN, NaN, NaN], NaN, false)
    end
end

function main()
    results_dir = joinpath(dirname(@__DIR__), "results")
    eval_path = joinpath(results_dir, "eval_results.csv")
    isfile(eval_path) || error("Eval results not found at $eval_path. Run the eval stage first.")

    df = CSV.read(eval_path, DataFrame)
    df = df[df.model_construction .== df.eval_construction, :]
    df.dose_value = [dose_value(c, d) for (c, d) in zip(df.model_construction, df.dose)]

    rows = DataFrame(construction=String[], seed=Int[], E0=Float64[], Emax=Float64[],
                     E50=Float64[], n=Float64[], r_squared=Float64[], converged=Bool[])

    for cons in CONSTRUCTIONS
        sub = df[df.model_construction .== cons, :]
        for s in sort(unique(sub.seed))
            cell = sort(sub[sub.seed .== s, :], :dose_value)
            if nrow(cell) < 4
                @warn "Skipping cell with too few dose points" construction=cons seed=s n=nrow(cell)
                continue
            end
            D = Float64.(cell.dose_value)
            Y = Float64.(cell.accuracy)
            params, r2, ok = fit_cell(D, Y)
            push!(rows, (cons, s, params[1], params[2], params[3], params[4], r2, ok))
        end
    end

    out_path = joinpath(results_dir, "hill_fits.csv")
    CSV.write(out_path, rows)
    println("Wrote $(nrow(rows)) Hill fits to $out_path")
end

main()
