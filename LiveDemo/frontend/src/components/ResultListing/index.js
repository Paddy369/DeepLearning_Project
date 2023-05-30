import { Card, CardMedia, CardContent } from '@mui/material'
import styles from './styles.module.css'

const ResultListing = ({ className, percentage, isMax }) => {

    console.log(className, isMax)

    return (
        <Card sx={{ display: 'flex', width: 250 }} className={`${styles.card} ${isMax && styles.max}`}>
            <CardMedia
                component="img"
                sx={{ width: 90 }}
                image={`/card_thumbnails/${className}.jpg`}
                alt="Cat"
            />
            <CardContent sx={{ flex: '1 0 auto' }} className={styles.contentWrapper}>
                <h3>{className}</h3>
                <div>Percentage: <b>{percentage}</b></div>
            </CardContent>
        </Card>
    )
}

export default ResultListing