const gulp = require("gulp");
const postcss = require("gulp-postcss");
const sourcemaps = require("gulp-sourcemaps");
const autoprefixer = require("autoprefixer");
const cssnano = require("cssnano");
const sass = require("gulp-sass")(require("sass"));
// const browserify = require("browserify");
// const nodeResolve = require("resolve");
// const source = require("vinyl-source-stream");
const uglify = require("gulp-uglify");
// const rename = require("gulp-rename");
// const buffer = require("vinyl-buffer");
// const babelify = require("babelify");
const concat = require("gulp-concat");
const {series, parallel} = require('gulp');

const paths = {
  scssPath: "./src/scss/*.scss",
  cssDest: "./static/css",
  faSrc: "./node_modules/@fontsource/roboto/files/*400*",
  fontDest: "./static/fonts",
  jsSrc: "./src/js/*.js",
  jsDest: "./static/js",
};

const vendorPaths = [
  "./node_modules/hyperscript.org/dist/_hyperscript_web.min.js",
  "./node_modules/jquery/dist/jquery.js",
  "./node_modules/htmx.org/dist/htmx.js",
  "./node_modules/bootstrap/dist/js/bootstrap.js",
  "./node_modules/select2/dist/js/select2.js",
  "./node_modules/plotly.js-cartesian-dist/plotly-cartesian.js",
];


function jsVendor() {
  return gulp
    .src(vendorPaths)
    .pipe(sourcemaps.init())
    .pipe(concat("vendors.js"))
    .pipe(uglify())
    .pipe(sourcemaps.write("."))
    .pipe(gulp.dest(paths.jsDest));
}

function jsApp() {
  return gulp
    .src(paths.jsSrc)
    .pipe(sourcemaps.init())
    .pipe(concat("App.js"))
    .pipe(uglify())
    .pipe(sourcemaps.write("."))
    .pipe(gulp.dest(paths.jsDest));
}

// compile scss into css
function scssTask() {
  return gulp
    .src(paths.scssPath) // Find scss file
    .pipe(sourcemaps.init())
    .pipe(sass({includePaths: ["./node_modules"]}).on("error", sass.logError))
    .pipe(
      postcss(
        [
          autoprefixer({env: "production"}),
          cssnano({
            preset: ["default", {discardComments: {removeAll: true}}],
          }),
        ],
        ""
      )
    )
    .pipe(sourcemaps.write("."))
    .pipe(gulp.dest(paths.cssDest)); // where to save the compiled css
}

function fonts() {
  return gulp.src(paths.faSrc).pipe(gulp.dest(paths.fontDest));
}

exports.style = scssTask;
exports.jsVendor = jsVendor;
exports.jsApp = jsApp;
exports.font = fonts;
exports.script = series(jsVendor, jsApp)
exports.assets = parallel(scssTask, series(jsVendor, jsApp), fonts)