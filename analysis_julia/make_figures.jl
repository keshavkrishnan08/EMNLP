# Julia mirror of the fig1 dose-response panel from src/drc/analysis/figures.py.
# Python is the reference implementation; this is provided for the PI's
# CairoMakie workflow and covers fig1 only (per-seed lines + mean per dose).
#
# Usage:  julia --project=analysis_julia analysis_julia/make_figures.jl
#
# Reads results/eval_results.csv and writes results/figures/fig1_dose_response_curves_julia.pdf

using CSV
using DataFrames
using Statistics
using CairoMakie

const CONSTRUCTIONS = ("aann", "comparative_correlative", "tough_movement", "resultative")
const DISPLAY_NAMES = Dict(
    "aann" => "AANN",
    "comparative_correlative" => "Comp. Correlative",
    "tough_movement" => "Tough Movement",
    "resultative" => "Resultative",
)
# Okabe-Ito colorblind-safe palette, one colour per construction.
const OKABE_ITO = ("#E69F00", "#56B4E9", "#009E73", "#F0E442")

const ATTESTED_ALL_COUNTS = Dict(
    "aann" => 1200,
    "comparative_correlative" => 340,
    "tough_movement" => 890,
    "resultative" => 2100,
)

function dose_value(construction, dose)
    if lowercase(strip(string(dose))) == "all"
        return float(ATTESTED_ALL_COUNTS[construction])
    end
    return parse(Float64, string(dose))
end

function main()
    results_dir = joinpath(dirname(@__DIR__), "results")
    eval_path = joinpath(results_dir, "eval_results.csv")
    isfile(eval_path) || error("Eval results not found at $eval_path. Run the eval stage first.")

    df = CSV.read(eval_path, DataFrame)
    df = df[df.model_construction .== df.eval_construction, :]
    df.dose_value = [dose_value(c, d) for (c, d) in zip(df.model_construction, df.dose)]

    fig = Figure(size=(900, 700))
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (idx, cons) in enumerate(CONSTRUCTIONS)
        r, c = positions[idx]
        ax = Axis(fig[r, c]; title=get(DISPLAY_NAMES, cons, cons),
                  xlabel="log10(dose + 1)", ylabel="SLOR accuracy")
        color = OKABE_ITO[mod1(idx, length(OKABE_ITO))]
        sub = df[df.model_construction .== cons, :]

        # Per-seed thin lines.
        for s in sort(unique(sub.seed))
            cell = sort(sub[sub.seed .== s, :], :dose_value)
            x = log10.(Float64.(cell.dose_value) .+ 1.0)
            lines!(ax, x, Float64.(cell.accuracy); color=(color, 0.3), linewidth=0.8)
        end

        # Mean per dose across seeds.
        doses = sort(unique(sub.dose_value))
        means = [mean(sub[sub.dose_value .== d, :accuracy]) for d in doses]
        xd = log10.(Float64.(doses) .+ 1.0)
        lines!(ax, xd, means; color=color, linewidth=2.5)
        scatter!(ax, xd, means; color=color, markersize=10)
        hlines!(ax, [0.5]; color=:gray, linestyle=:dot, linewidth=0.6)
        ylims!(ax, 0.0, 1.0)
    end

    fig_dir = joinpath(results_dir, "figures")
    mkpath(fig_dir)
    out_path = joinpath(fig_dir, "fig1_dose_response_curves_julia.pdf")
    save(out_path, fig)
    println("Wrote $out_path")
end

main()
