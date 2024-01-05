# Changelog

## [0.1.2](https://github.com/ockr-io/ockr-ocr-container/compare/v0.1.1...v0.1.2) (2024-01-05)


### Features

* add a proper resize logic to make sure the resize ratio is nearly the same for height and with but dividable by 32 ([83c2331](https://github.com/ockr-io/ockr-ocr-container/commit/83c23310bee5b4899355b4d31f86a19d2b52b0b5))
* add model name, model version and parameters to the response ([f18da7e](https://github.com/ockr-io/ockr-ocr-container/commit/f18da7e8be8b5945ba2729a06e0b196279ae34e8))
* include provided parameters in the response ([7b2c0eb](https://github.com/ockr-io/ockr-ocr-container/commit/7b2c0ebba3b56089cda2884e14e40ec6a5c952d0))
* support list of submodels within the ocr container ([28de4fc](https://github.com/ockr-io/ockr-ocr-container/commit/28de4fcfea8c746f555b249cff3ff50ff460d860))


### Bug Fixes

* add simple reshape logic to prevent bad ocr results on portrait images ([f347d22](https://github.com/ockr-io/ockr-ocr-container/commit/f347d2278a7868e31c7fff5b2f11b88864e5506c))

## [0.1.1](https://github.com/ockr-io/ockr-ocr-container/compare/v0.1.0...v0.1.1) (2023-12-15)


### Features

* download models from model ([cd68858](https://github.com/ockr-io/ockr-ocr-container/commit/cd68858144e036681fafe4ef4ce7e67f883b341d))
* implement the actual base64 input endpoint that does the ocr using a model from the ockr model zoo ([fc4bc7e](https://github.com/ockr-io/ockr-ocr-container/commit/fc4bc7ed7569e51f73bc9f7b6824698dd82992dd))

## 0.1.0 (2023-12-09)


### Features

* add basic project setup and endpoints ([b30a129](https://github.com/ockr-io/ockr-ocr-container/commit/b30a12907b331f8365b12816b5360e6f21fd41d7))
