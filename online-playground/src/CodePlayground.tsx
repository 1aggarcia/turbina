import { Editor } from "@monaco-editor/react";
import { CSSProperties, useState } from "react";
import { run_turbina_program } from "turbina";

const SAMPLE_PROGRAM =
`// define a binding like this
let x: int = 5;

// the type can be inferred
let y = "this is a string";

// 'let' bindings cannot be reassigned
// let x = 20;  // ERROR

// define functions like this
let square(x: int) -> x * x;
square(5);

// or do it the long way
let other_square = (x: int) -> x * x;

// all if statements are expressions
let isPalindrome(str: string) ->
    if (str == reverse(str)) "yes" else "no";

isPalindrome("racecar");
isPalindrome("palindrome");

// recursive functions require explicit return types, normal functions don't
let factorial(n: int): int ->
    if (n == 0) 1
    else        n * factorial(n - 1);

factorial(0);
factorial(5);
factorial(10);
`

const CODE_FONT_SIZE = 13;

const containerStyle: CSSProperties = {
    fontSize: CODE_FONT_SIZE,
    height: "100vh",
    width: "100%",
};

const outputDivStyle: CSSProperties = {
    flexGrow: 1,
    textAlign: "left",
    padding: "1rem",
    border: "1px solid rgb(192, 192, 192)",
};

/** Should not be mounted until the web assembly has been initialized */
export function CodePlayground() {
    const [sourceCode, setSourceCode] = useState(SAMPLE_PROGRAM);
    const [output, setOutput] = useState({
        results: [] as string[],
        isError: false,
    });

    const outputColor = output.isError ? "red" : "black";

    function onExecuteClick() {
        let output;
        try {
            output = {
                results: run_turbina_program(sourceCode),
                isError: false,
            };
        } catch (error) {
            output = {
                results: [`${error}`],
                isError: true,
            };
            console.error(error);
        }
        setOutput(output);
    }

    function onClearClick() {
        setOutput({ results: [], isError: false });
    }

    return (
        <div style={containerStyle}>
            <div>
                <button onClick={onExecuteClick}>Execute</button>
                <button onClick={onClearClick}>Clear</button>
            </div>

            <div style={{ display: "flex", height: "100%" }}>
                <Editor
                    value={sourceCode}
                    /* Rust syntax highlighting works decently and doesn't give
                     * validation errors */
                    language="rust"
                    onChange={text => setSourceCode(text ?? "")}
                    options={{
                        minimap: { enabled: false },
                        fontSize: CODE_FONT_SIZE,
                        contextmenu: false,
                        scrollBeyondLastLine: false,
                    }}
                    width={"60%"}
                />
                <div style={{ width: "8px", backgroundColor: "#eaeaea" }} />
                <div style={{...outputDivStyle, color: outputColor}}>
                    {output.results.map((r) => (
                        <p><code>{r}</code></p>
                    ))}
                </div>
            </div>
        </div>
    )
}
