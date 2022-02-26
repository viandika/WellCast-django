const Path = require("path");
const Webpack = require("webpack");
const { merge } = require("webpack-merge");
const ESLintPlugin = require("eslint-webpack-plugin");
const StylelintPlugin = require("stylelint-webpack-plugin");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");

const common = require("./webpack.common.js");

module.exports = merge(common, {
  target: "web",
  mode: "development",
  devtool: "inline-cheap-source-map",
  output: {
    chunkFilename: "js/[name].chunk.js",
    publicPath: "http://localhost:9247/",
  },
  devServer: {
    port: 9247,
    hot: true,
    webSocketServer: "ws",
    headers: {
      "Access-Control-Allow-Origin": "*",
    },
    devMiddleware: {
      writeToDisk: true,
    },
  },
  plugins: [
    new Webpack.DefinePlugin({
      "process.env.NODE_ENV": JSON.stringify("development"),
    }),
    new ESLintPlugin({
      extensions: "js",
      emitWarning: true,
      files: Path.resolve(__dirname, "./src"),
      fix: true,
    }),
    new StylelintPlugin({
      files: Path.join("src", "**/*.s?(a|c)ss"),
      fix: true,
    }),
    new MiniCssExtractPlugin({ filename: "css/app.css" }),
  ],
  module: {
    rules: [
      {
        test: /.*plotly.*\.js$/,
        loader: "ify-loader",
      },
      {
        test: /\.js$/,
        include: Path.resolve(__dirname, "./src"),
        loader: ESLintPlugin.loader,
      },
      {
        test: /\.s?css$/i,
        use: [
          MiniCssExtractPlugin.loader,
          {
            loader: "css-loader",
            options: {
              sourceMap: true,
            },
          },
          // "postcss-loader",
          "sass-loader",
        ],
      },
    ],
  },
});
