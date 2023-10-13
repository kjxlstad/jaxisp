from jaxtyping import install_import_hook

with install_import_hook("jaxisp", "beartype.beartype"):
    from jaxisp import nodes
    from jaxisp import helpers
