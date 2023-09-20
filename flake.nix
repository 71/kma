{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  inputs.flake-parts.url = "github:hercules-ci/flake-parts";

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "aarch64-darwin" "aarch64-linux" "x86_64-darwin" "x86_64-linux" ];

      perSystem = { config, pkgs, ... }:
        let
          python3 = pkgs.python311.withPackages (ps: [
            ps.numpy
            ps.torch
            ps.dill
            ps.tqdm
            ps.pyyaml
            (ps.buildPythonPackage rec {
              pname = "torchtext";
              version = "0.6.0";
              src = ps.fetchPypi {
                inherit pname version;
                sha256 = "sha256-dSL7tY6EfWmPbGoyU9TJSC4ZNgj6DKZHxc/JfWMFGMQ=";
              };
              doCheck = false;
            })
          ]);

          models = pkgs.stdenv.mkDerivation {
            name = "models";
            phases = [ "unpackPhase" "installPhase" ];
            src = pkgs.fetchzip {
              url = "https://docs.google.com/uc?export=download&id=1uzOEZnyqlz0EJNWMDdXMCeTHecmfZg7S";
              hash = "sha256-w5Gj2cqISUps1n1KjBEMjGt7hyO5GYj9YH5EXUS0oUk=";
              extension = "zip";
              stripRoot = false;
            };
            installPhase = ''
              mkdir -p $out
              cp $src/*.pt $out
            '';
          };
        in {
          devShells.default = pkgs.mkShell {
            nativeBuildInputs = [ python3 ];
          };

          packages.train = pkgs.writeShellScriptBin "train" ''
            ${python3}/bin/python3 train.py "$@"
          '';

          packages.tag = pkgs.writeShellScriptBin "tag" ''
            cp -n ${models}/*.pt model/
            ${python3}/bin/python3 tagging.py "$@"
          '';
        };
    };
}
