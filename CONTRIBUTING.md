## Contributing guidelines

Contributors are welcome to submit their code and ideas. In a long run, we hope this project can be managed by developers from both inside and outside Alibaba.

### Contributor License Agreements

* Sign CLA of MetaDist:
  Please download MetaDist [CLA](https://gist.github.com/alibaba-oss/151a13b0a72e44ba471119c7eb737d74). Follow the instructions to sign it.

### Pull Request Checklist

Here is a checklist to prepare and submit your PR (pull request).

* Create your own Github branch by forking MetaDist.
* Read the [README](README.md).
* Read the [contributing guidelines](CONTRIBUTING.md).
* Read the [Code of Conduct](CODE_OF_CONDUCT.md).
* Ensure you have signed the
  [Contributor License Agreement (CLA)](https://gist.github.com/alibaba-oss/151a13b0a72e44ba471119c7eb737d74).
* Push changes to your personal fork.
* Create a PR with a detail description, if commit messages do not express themselves.
* Submit PR for review and address all feedbacks.
* Wait for merging (done by committers).

Let's use an example to walk through the list.

## An Example of Submitting Code Change to MetaDist

### Fork Your Own Branch

On Github page of [MetaDist](https://github.com/alibaba/easydist), Click **fork** button to create your own easydist repository.

### Create Local Repository
```bash
git clone --recursive https://github.com/your_github/easydist.git
```
### Create a dev Branch (named as your_github_id_feature_name)
```bash
git branch your_github_id_feature_name
```
### Make Changes and Commit Locally
```bash
git status
git add files-to-change
git commit -m "messages for your modifications"
```

### Rebase and Commit to Remote Repository
```bash
git checkout main
git pull
git checkout your_github_id_feature_name
git rebase main
-- resolve conflict, run test --
git push --recurse-submodules=on-demand origin your_github_id_feature_name
```

### Create a PR
Click **New pull request** or **Compare & pull request** button, choose to compare branches easydist/main and your_github/your_github_id_feature_name, and write PR description.

### Address Reviewers' Comments
Resolve all problems raised by reviewers and update PR.

### Merge
It is done by MetaDist committers.
___

Copyright © Alibaba Group, Inc.
