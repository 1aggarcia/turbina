import { CSSProperties, PropsWithChildren } from "react";

const headerStyle: CSSProperties = {
    display: "flex",
    justifyContent: "space-evenly",
    alignItems: "center",
    marginBottom: "8px",
    marginLeft: "10px",
    marginRight: "10px",
}

/** Renders children in the center of the header */
export function Header(props: PropsWithChildren) {
    return (
        <div style={headerStyle}>
            <h2 style={{ flex: 1, margin: 0 }}>
                <a href="https://github.com/1aggarcia/turbina">Turbina Playground</a>
            </h2>

            <div>{props.children}</div>

            <p
                className="text-secondary"
                style={{ flex: 1, textAlign: "right", margin: 0 }}
            >
                Ripoff of TypeScript playground with LeetCode styling
            </p>
        </div>
    )
}
