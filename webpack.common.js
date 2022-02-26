const Path = require("path");
const { CleanWebpackPlugin } = require("clean-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const BundleTracker = require("webpack-bundle-tracker");
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");

module.exports = {
  entry: {
    app: Path.resolve(__dirname, "./src/js/main.js"),
  },
  output: {
    path: Path.join(__dirname, "./static"),
    filename: "js/[name].js",
    publicPath: "/static/",
  },
  optimization: {
    splitChunks: {
      chunks: "all",
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name(module) {
            const packageName = module.context.match(
              /[\\/]node_modules[\\/](.*?)([\\/]|$)/
            )[1];
            return `npm.${packageName.replace("@", "")}`;
          },
        },
      },
    },
  },
  plugins: [
    new CleanWebpackPlugin(),
    new NodePolyfillPlugin(),
    new CopyWebpackPlugin({
      patterns: [{ from: Path.resolve(__dirname, "./src/img"), to: "img" }],
    }),
    // new HtmlWebpackPlugin({
    //   template: Path.resolve(__dirname, "./src/index.html"),
    // }),
    new BundleTracker({ filename: "./webpack-stats.json" }),
  ],
  resolve: {
    alias: {
      "~": Path.resolve(__dirname, "../src"),
    },
  },
  module: {
    rules: [
      {
        test: /\.mjs$/,
        include: /node_modules/,
        type: "javascript/auto",
      },
      // {
      //   test: /\.html$/i,
      //   loader: "html-loader",
      // },
      {
        test: /\.(ico|jpg|jpeg|png|gif|eot|otf|webp|svg)(\?.*)?$/,
        type: "asset",
      },
      {
        test: /\.(ttf|woff|woff2)?$/,
        type: "asset",
        generator: {
          filename: "fonts/[name][ext]",
        },
      },
    ],
  },
};
