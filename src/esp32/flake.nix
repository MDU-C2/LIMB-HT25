{
  description = "LIMB Bionic Arm ESP32-C3 development tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
    esp-dev = {
      url = "github:mirrexagon/nixpkgs-esp-dev";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    , esp-dev
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ esp-dev.overlays.default ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.clang-tools
            pkgs.esp-idf-full
          ];

          shellHook = ''
            export CLANGD_QUERY_DRIVER=`which xtensa-esp32-elf-gcc`
            export IDF_TOOLCHAIN='gcc'
          '';
        };
      }
    );
}
