import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Classification from './routes/Classification';

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<Classification />}></Route>
            </Routes>
        </Router>
    )
}

export default App;
