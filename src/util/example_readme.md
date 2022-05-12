# Software Metadata Extraction Framework (SOMEF) 
https://pypi.org/project/somef/

```
cd somef
pip install -e .
```

<img src="docs/logo.png" alt="logo" width="150"/>

A command line interface for automatically extracting relevant information from readme files.

## Features
Given a readme file (or a GitHub/Gitlab repository) SOMEF will extract the following categories (if present):
- **Name**: Name identifying a software component
- **Full name**: Name + owner (owner/name)
- **Source code**: Link to the source code (typically the repository where the readme can be found)

If you don't include an authentication token, you can still use SOMEF.

ディープラーニングに関する論文の実装を先行研究から順に進める