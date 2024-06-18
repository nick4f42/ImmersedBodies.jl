using ImmersedBodies
using Documenter

DocMeta.setdocmeta!(ImmersedBodies, :DocTestSetup, :(using ImmersedBodies); recursive=true)

makedocs(;
    modules=[ImmersedBodies],
    authors="Nick OBrien <nick4f42@proton.me> and contributors",
    sitename="ImmersedBodies.jl",
    format=Documenter.HTML(;
        canonical="https://nick.github.io/ImmersedBodies.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/nick/ImmersedBodies.jl", devbranch="main")
