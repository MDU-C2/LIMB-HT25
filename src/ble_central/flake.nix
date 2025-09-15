{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default =
          pkgs.mkShell {
            buildInputs = with pkgs; [
              python313Packages.bleak
            ];
          };
      }
    );
}
