# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2024-05-29

### Added
- Trim trailing whitespace from input rows; increase robustness.

## [0.2.0] - 2024-05-29

### Added
- Read problems also from standard input.
- Add feature `mapgen`, which allows map generation with binary `generate-maze`.

### Changed
- Rename main binary executable to `solve-maze`
- Maze `hero_start`, `dragon_start` and `goal` are now usize indexes.
- Errors are separated for cases: no viable paths to end, or no paths without meeting the dragon.
- If there is no viable path from the dragon position to the hero position, the problem can be solved but the dragon does not move.

## [0.1.2] - 2024-05-28

### Changed
- Dijkstra's algorithm distance takes into account the dragon position

### Fixed
- Dragon movement tracks current hero position, previously it was n-1.

## [0.1.1] - 2024-05-28

### Changed
- Correct readme.

## [0.1.0] - 2024-05-28

### Added

- Command line program for solving a maze from a file.
- Hero routing based on shortest path algorithm.
- Dragon routing based on shortest path to current hero position.
- Playback solution.

[unreleased]: https://github.com/mkouhia/wundernut-vol13/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/mkouhia/wundernut-vol13/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mkouhia/wundernut-vol13/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/mkouhia/wundernut-vol13/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/mkouhia/wundernut-vol13/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mkouhia/wundernut-vol13/tree/v0.1.0