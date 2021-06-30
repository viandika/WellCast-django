const gulp = require('gulp');
const sass = require('gulp-sass');
const sourcemaps = require('gulp-sourcemaps');
const postcss = require('gulp-postcss');
const autoprefixer = require('autoprefixer');
const cssnano = require('cssnano');
const path = require('path');


const paths = {
  scssPath: './src/scss/**/*.scss',
  cssDest: './static/css',
  faSrc: './node_modules/@fortawesome/fontawesome-free/webfonts/*',
  fontDest: './static/webfonts'
}


// compile scss into css
function scssTask() {
  return gulp.src(paths.scssPath) // Find scss file
    .pipe(sourcemaps.init()) // initialize sourcemaps
    .pipe(sass({
      precision: 7,
      importer: (url, prev, done) => {
        if (url[0] === '~') {
          url = path.resolve('node_modules', url.substr(1));
        }

        return {file: url};
      },
    }).on('error', sass.logError)) // pass scss through sass compiler
    .pipe(postcss([autoprefixer(), cssnano()])) // PostCSS plugins
    .pipe(sourcemaps.write('.')) // write sourcemaps file in current directory
    .pipe(gulp.dest(paths.cssDest)) // where to save the compile css
}

function webfonts() {
  return gulp.src(paths.faSrc)
    .pipe(gulp.dest(paths.fontDest))
}


exports.style = scssTask;
exports.font = webfonts;
