import loadWasm from "turbina";
import { useEffect, useState } from 'react'

function App() {
    const [isLoaded, setIsLoaded] = useState(false);
    const [error, setError] = useState<null | Error>(null);

    useEffect(() => {
        loadWasm()
            .then(() => setIsLoaded(true))
            .catch(err => setError(err));
    }, []);

    if (error !== null) {
        return <p>ERROR LOADING WEB ASSEMBLY: {error.toString()}</p>
    }
    if (!isLoaded) {
        return <p>Loading Web Assembly...</p>
    }
    return <p>Web Assembly loaded successfully</p>
}

export default App;
