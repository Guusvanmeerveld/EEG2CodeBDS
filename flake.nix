{
  description = "";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};

      python = pkgs.python3;

      pyntbci =
        python.pkgs.buildPythonPackage
        {
          pname = "pyntbci";
          version = "1.8.4";
          format = "pyproject";

          nativeBuildInputs =
            (with pkgs; [
              pkg-config
            ])
            ++ (with python.pkgs; [
              setuptools
              scikit-learn
              h5py
              matplotlib
              mne
              numpy
              scipy
              seaborn
            ]);

          src = pkgs.fetchFromGitHub {
            owner = "thijor";
            repo = "pyntbci";
            rev = "4072ed0be749b40f858528e864f9ea32c7cef7fd";
            hash = "sha256-MoDCNS7FCA/RXDmDKIjWZuiilkv8gQCqhg6Me0nBBvo=";
          };
        };

      pythonWithPkgs = python.withPackages (ps:
        (with ps; [
          scipy
          scikit-learn
          keras
          matplotlib
          seaborn
          mne
        ])
        ++ [
          pyntbci
        ]);
    in {
      devShells.default = pkgs.mkShell {
        buildInputs = [
          pythonWithPkgs
        ];
      };
    });
}
