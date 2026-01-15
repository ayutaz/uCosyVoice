# GitHub Actions Setup Guide

This project uses [GameCI](https://game.ci/) for Unity CI/CD.

## Required Secrets

You need to configure the following secrets in your GitHub repository settings:

### Option 1: Unity License File (Recommended)

1. **UNITY_LICENSE**: Your Unity license file content (`.ulf` file)

To get your license file:

```bash
# Run Unity activation locally
docker run -it --rm \
  -e UNITY_VERSION=6000.0.58f2 \
  -e UNITY_EMAIL=your@email.com \
  -e UNITY_PASSWORD=yourpassword \
  unityci/editor:ubuntu-6000.0.58f2-base-3 \
  unity-editor -quit -batchmode -nographics \
  -logFile /dev/stdout \
  -createManualActivationFile
```

Then activate manually at [license.unity3d.com](https://license.unity3d.com/) and add the resulting `.ulf` file content as the `UNITY_LICENSE` secret.

### Option 2: Unity Credentials

- **UNITY_EMAIL**: Your Unity account email
- **UNITY_PASSWORD**: Your Unity account password

Note: This option may not work with 2FA enabled.

## Workflows

### unity.yml

Main CI workflow that runs on push and pull requests:

1. **Test Job**: Runs all EditMode tests
2. **Build Job**: Creates Windows (Mono) build (only runs if tests pass)

### Triggers

- Push to `main`, `develop`, or `feature/**` branches
- Pull requests to `main` or `develop`
- Manual trigger via workflow_dispatch

## Artifacts

- **Test Results**: Test output and coverage reports
- **Build-Windows-Mono**: Windows executable build

## Important: ONNX Models

ONNX model files (`.onnx`) are **not tracked in git** due to their large size (~4GB total).

### Impact on CI

- **Tests**: Model-dependent tests will fail without ONNX files
- **Build**: Compilation will succeed, but runtime requires models

### Solutions

1. **Git LFS** (Recommended for private repos):
   ```bash
   git lfs install
   git lfs track "*.onnx"
   git add .gitattributes
   git add Assets/Models/*.onnx
   git commit -m "Add ONNX models with LFS"
   ```

2. **External Storage**: Download models in CI workflow from cloud storage (S3, GCS, etc.)

3. **Skip Model Tests**: Add `[Category("RequiresModels")]` attribute to tests and exclude in CI

### Current Workflow Behavior

The workflow will attempt to run all tests. Tests that require ONNX models may fail if models are not available. This is expected behavior for repositories without model files.

## Notes

- Consider using Git LFS for large model files if full CI testing is needed
- The build artifact will not include ONNX models unless they are tracked
