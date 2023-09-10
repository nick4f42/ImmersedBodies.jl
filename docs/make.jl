using Documenter
using ImmersedBodies

makedocs(
    sitename="ImmersedBodies",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Examples" => [
            "examples/naca.md",
        ],
        "Reference" => [
            "reference/problems.md",
            "reference/solving.md",
        ],
    ],
)

deploydocs(; repo="github.com/NUFgroup/ImmersedBodies.jl.git")
