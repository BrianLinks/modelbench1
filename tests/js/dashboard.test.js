'use strict';
// Run with: node --test tests/js
const test = require('node:test');
const assert = require('node:assert/strict');
const ui = require('../../src/modelbench/static/dashboard.js');

test('parseMetrics reads name=value pairs and ignores anything else', () => {
  assert.deepEqual(ui.parseMetrics('accuracy=0.9000 f1=0.8000'), { accuracy: 0.9, f1: 0.8 });
  assert.deepEqual(ui.parseMetrics('  loss=1e-3   r2=-0.5 '), { loss: 0.001, r2: -0.5 });
  assert.deepEqual(ui.parseMetrics('no metrics here'), {});
  assert.deepEqual(ui.parseMetrics('a=NaN b=inf c= =3 d=1x'), {});
  assert.deepEqual(ui.parseMetrics('fold/accuracy=0.5'), { 'fold/accuracy': 0.5 });
  assert.deepEqual(ui.parseMetrics('a=b=2'), { 'a=b': 2 });
  assert.deepEqual(ui.parseMetrics(''), {});
});

test('sortOrder sorts numbers ascending and descending, keeping ties in place', () => {
  assert.deepEqual(ui.sortOrder([3, 1, 2], true, false), [1, 2, 0]);
  assert.deepEqual(ui.sortOrder([3, 1, 2], true, true), [0, 2, 1]);
  assert.deepEqual(ui.sortOrder([1, 1, 0, 1], true, false), [2, 0, 1, 3]);
  assert.deepEqual(ui.sortOrder([1, 1, 0, 1], true, true), [0, 1, 3, 2]);
  assert.deepEqual(ui.sortOrder([], true, false), []);
});

test('sortOrder puts missing values last in both directions', () => {
  const values = [null, 2, NaN, 1, undefined, ''];
  assert.deepEqual(ui.sortOrder(values, true, false), [3, 1, 0, 2, 4, 5]);
  assert.deepEqual(ui.sortOrder(values, true, true), [1, 3, 0, 2, 4, 5]);
});

test('sortOrder compares text by code unit, not by locale', () => {
  assert.deepEqual(ui.sortOrder(['b', 'B', 'a', 'run-10', 'run-2'], false, false), [1, 2, 0, 3, 4]);
  assert.deepEqual(ui.sortOrder(['b', 'a'], false, true), [0, 1]);
});

test('sortOrder does not change its input', () => {
  const values = [3, 1, 2];
  ui.sortOrder(values, true, false);
  assert.deepEqual(values, [3, 1, 2]);
});

test('rowMatches needs every term, ignores case, and matches everything for an empty query', () => {
  assert.equal(ui.rowMatches('run-000001 KNN completed', 'knn'), true);
  assert.equal(ui.rowMatches('run-000001 KNN completed', 'KNN run-000001'), true);
  assert.equal(ui.rowMatches('run-000001 KNN completed', 'knn failed'), false);
  assert.equal(ui.rowMatches('run-000001', ''), true);
  assert.equal(ui.rowMatches('run-000001', '   '), true);
  assert.equal(ui.rowMatches('', 'x'), false);
  assert.equal(ui.rowMatches('a  b', 'a b'), true);
});

test('nearestIndex picks the closest point and honours the limit', () => {
  const points = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }];
  assert.equal(ui.nearestIndex(points, 9, 1), 1);
  assert.equal(ui.nearestIndex(points, 9, 9), 2);
  assert.equal(ui.nearestIndex(points, 100, 100, 5), -1);
  assert.equal(ui.nearestIndex(points, 3, 4, 5), 0); // exactly at the limit counts
  assert.equal(ui.nearestIndex(points, 3, 4, 4.99), -1);
  assert.equal(ui.nearestIndex([], 0, 0), -1);
  assert.equal(ui.nearestIndex([{ x: 1, y: 1 }, { x: 1, y: 1 }], 1, 1), 0); // a tie keeps the first
});

test('segmentDistance is measured to the segment, not the infinite line', () => {
  assert.equal(ui.segmentDistance(5, 3, 0, 0, 10, 0), 3); // above the middle
  assert.equal(ui.segmentDistance(-3, 4, 0, 0, 10, 0), 5); // beyond the start: distance to (0, 0)
  assert.equal(ui.segmentDistance(13, 4, 0, 0, 10, 0), 5); // beyond the end
  assert.equal(ui.segmentDistance(3, 4, 0, 0, 0, 0), 5); // a zero-length segment is a point
  assert.equal(ui.segmentDistance(5, 0, 0, 0, 10, 0), 0);
});

test('polylineDistance covers every segment and single points', () => {
  const line = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }];
  assert.equal(ui.polylineDistance(line, 12, 5), 2);
  assert.equal(ui.polylineDistance(line, 5, -1), 1);
  assert.equal(ui.polylineDistance([{ x: 1, y: 1 }], 4, 5), 5);
  assert.equal(ui.polylineDistance([], 0, 0), Infinity);
});

test('nearestLine returns the closest line within the limit, later lines winning ties', () => {
  const lines = [
    [{ x: 0, y: 0 }, { x: 10, y: 0 }],
    [{ x: 0, y: 10 }, { x: 10, y: 10 }],
    [{ x: 5, y: 5 }],
  ];
  assert.equal(ui.nearestLine(lines, 5, 1, 3), 0);
  assert.equal(ui.nearestLine(lines, 5, 9, 3), 1);
  assert.equal(ui.nearestLine(lines, 5, 4, 3), 2);
  assert.equal(ui.nearestLine(lines, 5, 5, 0), 2);
  assert.equal(ui.nearestLine(lines, 5, 30, 3), -1);
  assert.equal(ui.nearestLine(lines, 20, 0, 3), -1);
  assert.equal(ui.nearestLine([], 0, 0, 3), -1);
  const tied = [[{ x: 0, y: 0 }, { x: 10, y: 0 }], [{ x: 0, y: 0 }, { x: 10, y: 0 }]];
  assert.equal(ui.nearestLine(tied, 5, 1, 3), 1);
});

test('parsePoints reads an SVG points attribute and skips malformed pairs', () => {
  assert.deepEqual(ui.parsePoints('1.5,2 3,4.25'), [{ x: 1.5, y: 2 }, { x: 3, y: 4.25 }]);
  assert.deepEqual(ui.parsePoints('  1,2   3,4  '), [{ x: 1, y: 2 }, { x: 3, y: 4 }]);
  assert.deepEqual(ui.parsePoints('1,2 oops 3 4,x 5,6,7 8,9'), [{ x: 1, y: 2 }, { x: 8, y: 9 }]);
  assert.deepEqual(ui.parsePoints(''), []);
});

test('cellText shows the count or the rounded percent of the row', () => {
  assert.equal(ui.cellText(3, 0.75, 'count'), '3');
  assert.equal(ui.cellText(3, 0.75, 'percent'), '75%');
  assert.equal(ui.cellText(0, 0, 'percent'), '0%');
  assert.equal(ui.cellText(1, 0.3333, 'percent'), '33%');
  assert.equal(ui.cellText(1, 0.005, 'percent'), '1%');
  assert.equal(ui.cellText(7, 1, 'percent'), '100%');
});

test('isRunPath accepts only run pages', () => {
  assert.equal(ui.isRunPath('/run/run-000001'), true);
  assert.equal(ui.isRunPath('/run/a.b_c-1'), true);
  for (const bad of ['/run/', '/run/a/b', '//evil.example/run/x', 'https://evil.example/run/x',
    '/run/x?y=1', '/run/x#y', 'javascript:alert(1)', '/api/run/x', '/run/' + 'a'.repeat(65), '',
    '/run/a b', '/run/<x>', '/run/..%2f']) {
    assert.equal(ui.isRunPath(bad), false, bad);
  }
  assert.equal(ui.isRunPath('/run/' + 'a'.repeat(64)), true);
});
