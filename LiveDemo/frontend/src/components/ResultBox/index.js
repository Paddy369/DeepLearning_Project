import { useState, useEffect } from 'react'
import ResultListing from '../ResultListing'
import styles from './styles.module.css'

const ResultBox = ({image, milestone, model, state, setState}) => {

    console.log("box called")

    const [max, setMax] = useState("")

    const [results, setResults] = useState({
        "Beetle": null, "Butterfly": null, "Cat": null, "Cow": null, "Dog": null, "Elephant": null, "Gorilla": null, 
        "Hippo": null, "Lizard": null, "Monkey": null, "Mouse": null, "Panda": null, "Spider": null, "Tiger": null, "Zebra": null
    })

    const getMax = results => {
        return Object.entries(results).reduce((prevEntry, currEntry) => prevEntry[1] > currEntry[1] ? prevEntry : currEntry)
    }

    useEffect(() => {
        console.log("effect called")
        async function fetchData() {
            const res = await fetchResults()
            console.log("result box ", res)
            setResults(res)
            setMax(getMax(res)[0])
            setState(Object.keys(res).length > 0 ? "loaded" : "")
        }

        if(image && milestone && model)
            fetchData()
    }, [image])

    const fetchResults = async () => {
        const res = await fetch(`http://localhost:5000/api/classify?model=${model}&milestone=${milestone}&image=${image}`)
        return res.json()
    }

    return (
        <div className={styles.wrapper}>
            {
                Object.entries(results).map(([key, value], idx) => 
                    <ResultListing key={idx} className={key} percentage={value} isMax={max == key} state={state} />
                )
            }
        </div>
    )
}

export default ResultBox;