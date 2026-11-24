using LUSOL
using Documenter

DocMeta.setdocmeta!(LUSOL, :DocTestSetup, :(using LUSOL); recursive=true)

makedocs(;
    modules=[LUSOL],
    authors="Thomas Leveringhaus",
    sitename="LUSOL.jl",
    format=Documenter.HTML(;
        canonical="https://100110110101.github.io/LUSOL.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/100110110101/LUSOL.jl",
    devbranch="master",
)
