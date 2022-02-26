import "../scss/style.scss";

import "hyperscript.org";
import $ from "jquery";

export { $ };

import "@popperjs/core";
import bootstrap from "bootstrap";
import "htmx.org/dist/htmx.js";
import * as Plotly from "plotly.js/lib/index-cartesian";

window.Plotly = Plotly;

import "select2";
import "./select-custom";
import "./index";
