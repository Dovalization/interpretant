import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";
import prettier from "eslint-config-prettier";
import reactHooks from "eslint-plugin-react-hooks";
import simpleImportSort from "eslint-plugin-simple-import-sort";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  prettier,
  globalIgnores([".next/**", "out/**", "build/**", "next-env.d.ts"]),
  {
    plugins: {
      "react-hooks": reactHooks,
      "simple-import-sort": simpleImportSort,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "simple-import-sort/imports": [
        "error",
        {
          groups: [
            // React and Next.js framework
            ["^react", "^next"],
            // External packages
            ["^@?\\w"],
            // Internal @/ aliases
            ["^@/"],
            // Type imports
            ["^.*\\u0000$"],
          ],
        },
      ],
      "simple-import-sort/exports": "error",
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", fixStyle: "inline-type-imports" },
      ],
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-non-null-assertion": "error",
      "no-restricted-syntax": [
        "error",
        {
          selector: "VariableDeclaration[kind='let']",
          message:
            "Use const and immutable flow. If mutation is genuinely required, isolate it behind a const-owned object/helper.",
        },
      ],
    },
  },
]);

export default eslintConfig;
