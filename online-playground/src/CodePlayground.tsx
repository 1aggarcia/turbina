import { Editor } from "@monaco-editor/react";
import { CSSProperties, useState } from "react";
import { run_turbina_program } from "turbina";
import { Header } from "./Header";
import { ResizablePanels } from "./ResizablePanels";

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
`;

const CODE_FONT_SIZE = 13;

const containerStyle: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    fontSize: CODE_FONT_SIZE,
    height: "100%",
    width: "100%",
};

const consoleDivStyle: CSSProperties = {
    fontSize: CODE_FONT_SIZE,
    textAlign: "left",
    padding: "1rem",
};

/** Should not be mounted until the web assembly has been initialized */
export function CodePlayground() {
    const [sourceCode, setSourceCode] = useState(SAMPLE_PROGRAM);
    const [consoleOutput, setConsoleOutput] = useState("");

    function writeToConsole(data: string) {
        // state change must be queued with a callback since the WASM runtime
        // doesn't wait for React updates
        setConsoleOutput(output => output + data);
    }

    function onExecuteClick() {
        try {
            run_turbina_program(
                sourceCode, writeToConsole, writeToConsole);
            writeToConsole("\n");
        } catch (error) {
            writeToConsole(`${error}\n`);
            console.error(error);
        }
    }

    function onClearClick() {
        setConsoleOutput("");
    }

    return (
        <div style={containerStyle}>
            <Header>
                <button className="round-left" onClick={onExecuteClick}>
                    Execute
                </button>
                <button className="round-right" onClick={onClearClick}>
                    Clear Output
                </button>
            </Header>

            <ResizablePanels
                firstComponent={
                    <div style={{ paddingBlock: "10px", width: "100%" }}>
                        <Editor
                            value={sourceCode}
                            /* Rust syntax highlighting works decently and
                            doesn't give validation errors */
                            language="rust"
                            onChange={text => setSourceCode(text ?? "")}
                            options={{
                                minimap: { enabled: false },
                                fontSize: CODE_FONT_SIZE,
                                contextmenu: false,
                                scrollBeyondLastLine: false,
                            }}
                        />
                    </div>
                }
                secondComponent={
                    <pre style={{ ...consoleDivStyle, margin: 0 }}>
                        {consoleOutput}
                    </pre>
                }
            />
        </div>
    )
}
