import { useState } from 'react'
import styles from './styles.module.css'
import ImageInputBox from '../../components/ImageInputBox'
import ResultBox from '../../components/ResultBox'
import Navbar from '../../components/Navbar'

const Home = () => {

    const [milestone, setMilestone] = useState("Meilenstein3")
    const [model, setModel] = useState("")
    const [image, setImage] = useState(null)
    const [state, setState] = useState("")

    return (
        <div className={styles.wrapper}>
            <div className={styles.content}>
                <Navbar milestone={milestone} setMilestone={setMilestone} model={model} setModel={setModel} />
                <div className={styles.classificationWrapper}>
                    <ImageInputBox setImage={setImage} setState={setState} />
                    <ResultBox image={image} milestone={milestone} model={model} state={state} setState={setState} />
                </div>
            </div>
        </div>
    )
}

export default Home